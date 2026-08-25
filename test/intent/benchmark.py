"""intent/benchmark.py

Benchmark 100-case extractor benchmark for three methods:
 - Rule-based NER (`intent/ner_extractor.NERExtractor`)
 - PhoBERT token-classification model (pretrained saved under repo)
 - LLM extractor (`intent/llm_extractor.LLMExtractor`)

Requirements and behavior are driven by the repo owner's instructions:
 - Read an external JSON file (default: ./100.json). Do NOT modify it.
 - Run ALL 100 cases in the exact same order for all 3 methods.
 - Do NOT train any model or change extractor logic.
 - Measure per-case latency, aggregate P50/P95/avg, compute exact-match
     accuracy per-field, full-extraction accuracy, entity-level P/R/F1, cost
     (LLM tokens if available), and save detailed results + a markdown report.

Usage (from repo root):
    python -m intent.benchmark --cases ./100.json

If required resources are missing (cases file, phobert model dir, or LLM
API key), the script will clearly report and will still run available
methods (e.g., run only Rule-based + PhoBERT if LLM key absent).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from intent.extractor_base import ExtractedEntities
from intent.ner_extractor import NERExtractor
from intent.llm_extractor import LLMExtractor

# Cost assumptions (LLM tokens -> USD). Keep as estimate only.
LLM_COST_PER_1K_TOKENS = 0.05

# Repeat per-case inference for latency percentiles. Keep default 1 to
# avoid excessive LLM calls/cost during benchmark; user may increase.
REPEATS = 1

# ---------------------------------------------------------------------------
# FIX: ner_test_set_100.jsonl dùng gold schema KHÁC với output chuẩn hoá của
# cả 3 extractor (size: "lớn"/"to"/"nhỏ"/"vừa" thay vì "L"/"M"; quantity: số
# viết bằng CHỮ hoặc STRING số thay vì int; product: chữ thường/rút gọn,
# không giống tên chuẩn trong menu.json). So sánh == thô sẽ luôn False dù
# model trả đúng. Các hàm dưới đây CHUẨN HOÁ CẢ GOLD LẪN PRED trước khi so
# sánh - KHÔNG sửa nội dung ner_test_set_100.jsonl, chỉ sửa cách benchmark.py
# CHẤM ĐIỂM.
try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    import unicodedata as _ud

    def unidecode(text: str) -> str:  # fallback tối thiểu cho tiếng Việt
        text = text.replace("đ", "d").replace("Đ", "D")
        return "".join(c for c in _ud.normalize("NFD", text) if not _ud.combining(c))


def _fold(s: str) -> str:
    return unidecode(str(s)).strip().lower()


_SIZE_WORD_TO_CODE = {
    "l": "L", "lon": "L", "to": "L",
    "m": "M", "nho": "M", "vua": "M",
}


def normalize_size(value):
    """Trả về "L"/"M"/None. Chấp nhận cả code sẵn ("L","M") lẫn từ mô tả
    ("lớn","to","nhỏ","vừa"), không phân biệt hoa/thường/dấu."""
    if value is None:
        return None
    folded = _fold(value)
    return _SIZE_WORD_TO_CODE.get(folded)  # None nếu không nhận diện được


_NUMBER_WORD_TO_INT = {
    "mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5,
    "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
}


def normalize_quantity(value):
    """Trả về int hoặc None. Chấp nhận số dạng string ("2"), số thật (2),
    hoặc số đếm bằng chữ ("một", "hai", ...)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    folded = _fold(s)
    return _NUMBER_WORD_TO_INT.get(folded)  # None nếu không parse được


def normalize_product(value):
    """Fold bỏ dấu + lower + bỏ khoảng trắng thừa, để so khớp không phân
    biệt hoa/thường/dấu/khoảng trắng (vd. "Bạc Xỉu" == "bạc xỉu")."""
    if value is None:
        return None
    return _fold(value).replace(" ", "")
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    text: str
    gold: dict


@dataclass
class ModelCaseResult:
    text: str
    gold: dict
    pred: dict
    latencies: List[float]
    total_tokens: Optional[int]
    field_correct: Dict[str, bool]

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


@dataclass
class ModelReport:
    name: str
    cases: List[ModelCaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def full_match_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if all(c.field_correct.values())) / self.n

    @property
    def field_accuracy(self) -> Dict[str, float]:
        if not self.cases:
            return {"product": 0.0, "size": 0.0, "quantity": 0.0}
        return {
            f: sum(1 for c in self.cases if c.field_correct.get(f)) / self.n
            for f in ("product", "size", "quantity")
        }

    @property
    def all_latencies(self) -> List[float]:
        return [lat for c in self.cases for lat in c.latencies]

    @property
    def avg_latency(self) -> float:
        vals = self.all_latencies
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def p50_latency(self) -> float:
        vals = sorted(self.all_latencies)
        if not vals:
            return 0.0
        return statistics.median(vals)

    @property
    def p95_latency(self) -> float:
        vals = sorted(self.all_latencies)
        if not vals:
            return 0.0
        idx = max(0, int(round(0.95 * (len(vals) - 1))))
        return vals[idx]

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens or 0 for c in self.cases)

    @property
    def total_cost_usd(self) -> float:
        return (self.total_tokens / 1000.0) * LLM_COST_PER_1K_TOKENS


def read_cases(path: str) -> List[CaseRecord]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cases file not found: {path}")
    # support both JSON list files and JSONL (one JSON object per line)
    raw: List[dict]
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(2)
        f.seek(0)
        if path.lower().endswith(".jsonl") or first.strip().startswith("{"):
            raw = [json.loads(line) for line in f if line.strip()]
        else:
            raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Cases file must be a JSON list or JSONL of cases")

    if len(raw) != 100:
        raise ValueError(f"Expected 100 test cases, found {len(raw)}")

    # Identify keys present in case objects
    sample_keys = set(raw[0].keys())
    if "text" not in sample_keys:
        raise ValueError(f"Cases must include a 'text' field; sample keys: {sorted(sample_keys)}")

    # Determine which gold keys are present (strict: do not guess beyond these known names)
    product_key = None
    for k in ("product", "product_name"):
        if k in sample_keys:
            product_key = k
            break
    if product_key is None or "size" not in sample_keys or "quantity" not in sample_keys:
        raise ValueError(
            f"Cases must include gold fields 'product'|'product_name', 'size', 'quantity'. Found keys: {sorted(sample_keys)}"
        )

    # deduplicate by text if duplicates exist: keep first occurrence
    seen_texts = set()
    deduped: List[dict] = []
    duplicates_info: List[Tuple[int, Optional[int], str]] = []  # (orig_index, id, text)
    for idx, c in enumerate(raw, 1):
        t = c.get("text")
        if t in seen_texts:
            duplicates_info.append((idx, c.get("id"), t))
            continue
        seen_texts.add(t)
        deduped.append(c)
    if duplicates_info:
        print(f"Warning: found {len(duplicates_info)} duplicate cases by 'text'. Keeping first occurrences; removed later duplicates:")
        for ln, oid, t in duplicates_info:
            print(f"  - removed line {ln}, id={oid}, text={t}")
    raw = deduped

    cases: List[CaseRecord] = []
    for i, c in enumerate(raw, 1):
        t = c.get("text")
        if t is None:
            raise ValueError(f"Case #{i} missing 'text' field")
        # Ensure gold labels exist (explicitly present, can be null)
        if product_key not in c:
            raise ValueError(f"Case #{i} missing '{product_key}'")
        if "size" not in c:
            raise ValueError(f"Case #{i} missing 'size'")
        if "quantity" not in c:
            raise ValueError(f"Case #{i} missing 'quantity'")

        cases.append(CaseRecord(text=t, gold={"product": c.get(product_key), "size": c.get("size"), "quantity": c.get("quantity")}))

    return cases


def normalize_from_extracted(obj: Any) -> dict:
    """Normalize various extractor outputs into dict with keys product,size,quantity.
    Accepts ExtractedEntities, dict, or any object with attributes.
    Returns string values or None. Does NOT fuzzy-match or correct semantics."""
    if obj is None:
        return {"product": None, "size": None, "quantity": None}
    # If ExtractedEntities-like (has .product_name)
    if hasattr(obj, "product_name") or hasattr(obj, "quantity"):
        prod = getattr(obj, "product_name", None)
        size = getattr(obj, "size", None)
        qty = getattr(obj, "quantity", None)
        return {"product": prod, "size": size, "quantity": qty}
    if isinstance(obj, dict):
        # Accept either 'product' or 'product_name'
        prod = obj.get("product") if "product" in obj else obj.get("product_name")
        return {"product": prod, "size": obj.get("size"), "quantity": obj.get("quantity")}
    # Fallback: try attributes
    return {"product": getattr(obj, "product", None), "size": getattr(obj, "size", None), "quantity": getattr(obj, "quantity", None)}


def compare_fields(gold: dict, pred: dict) -> Dict[str, bool]:
    return {
        "product": normalize_product(gold.get("product")) == normalize_product(pred.get("product")),
        "size": normalize_size(gold.get("size")) == normalize_size(pred.get("size")),
        "quantity": normalize_quantity(gold.get("quantity")) == normalize_quantity(pred.get("quantity")),
    }


def classify_error(gold: dict, pred: dict) -> List[str]:
    """Return list of error labels for a single case (e.g., PRODUCT missing, PRODUCT partial, SIZE wrong, ...)."""
    errors: List[str] = []

    g_prod, p_prod = normalize_product(gold.get("product")), normalize_product(pred.get("product"))
    g_size, p_size = normalize_size(gold.get("size")), normalize_size(pred.get("size"))
    g_qty, p_qty = normalize_quantity(gold.get("quantity")), normalize_quantity(pred.get("quantity"))

    # PRODUCT
    if g_prod is None:
        pass
    else:
        if p_prod is None:
            errors.append("PRODUCT missing")
        elif p_prod != g_prod:
            # partial detection: substring relation after fold (e.g. gold "coldbrew"
            # is a shortened alias of the full menu name the extractor returns)
            if p_prod in g_prod or g_prod in p_prod:
                errors.append("PRODUCT partial")
            else:
                errors.append("PRODUCT wrong")

    # SIZE
    if g_size is None:
        pass
    else:
        if p_size is None:
            errors.append("SIZE missing")
        elif p_size != g_size:
            errors.append("SIZE wrong")

    # QUANTITY
    if g_qty is None:
        pass
    else:
        if p_qty is None:
            errors.append("QUANTITY missing")
        elif p_qty != g_qty:
            errors.append("QUANTITY wrong")

    return errors


def load_phobert_model(candidate_dirs: List[str]) -> Tuple[Any, Any]:
    """Try to find and load a saved phobert token-classification model and tokenizer.
    Returns (tokenizer, model). Raises FileNotFoundError if not found or ImportError if transformers missing."""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    except Exception as e:
        raise ImportError("transformers library is required for PhoBERT inference") from e

    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        # basic check: config.json or tokenizer files exist
        if os.path.exists(os.path.join(d, "config.json")) or os.path.exists(os.path.join(d, "tokenizer_config.json")):
            # load
            tokenizer = AutoTokenizer.from_pretrained(d)
            model = AutoModelForTokenClassification.from_pretrained(d)
            return tokenizer, model

    raise FileNotFoundError("Could not find a PhoBERT model directory in candidates: " + ",".join(candidate_dirs))


def run_phobert_inference(tokenizer, model, text: str) -> dict:
    """Run token-classification pipeline with aggregation and return normalized dict."""
    from transformers import pipeline

    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )
    groups = pipe(text)
    # groups: list of dict with keys 'entity_group','word'
    out = {"product": None, "size": None, "quantity": None}
    for g in groups:
        name = str(g.get("entity_group", "")).lower()
        word = g.get("word")
        if name in ("product", "product_name") and out["product"] is None:
            out["product"] = word
        if name == "size" and out["size"] is None:
            out["size"] = word
        if name == "quantity" and out["quantity"] is None:
            # try to coerce to int if numeric
            try:
                out["quantity"] = int(word)
            except Exception:
                out["quantity"] = word
    return out


def measure_model_on_cases(cases: List[CaseRecord], methods: List[str]) -> Dict[str, ModelReport]:
    reports: Dict[str, ModelReport] = {}

    # 1) Rule-based NER
    if "rule" in methods:
        rb = NERExtractor()
        rb_report = ModelReport(name="Rule-based")
        for case in cases:
            lat = []
            pred_obj = None
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                pred_obj = rb.extract(case.text)
                lat.append(time.perf_counter() - t0)
            pred = normalize_from_extracted(pred_obj)
            fc = compare_fields(case.gold, pred)
            rb_report.cases.append(ModelCaseResult(text=case.text, gold=case.gold, pred=pred, latencies=lat, total_tokens=0, field_correct=fc))
        reports["rule"] = rb_report

    # 2) PhoBERT NER - try to find model under ner_model/best_phobert_ner or ner_model
    # 2) PhoBERT NER
    if "phobert" in methods:
        phobert_candidates = [os.path.join(os.getcwd(), "ner_model", "best_phobert_ner"), os.path.join(os.getcwd(), "ner_model")]
        try:
            tokenizer, model = load_phobert_model(phobert_candidates)
            ph_report = ModelReport(name="PhoBERT")
            for case in cases:
                lat = []
                pred = None
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    pred = run_phobert_inference(tokenizer, model, case.text)
                    lat.append(time.perf_counter() - t0)
                # quantity may be string -> keep as-is but try int
                if isinstance(pred.get("quantity"), str):
                    try:
                        pred["quantity"] = int(pred["quantity"])
                    except Exception:
                        pass
                fc = compare_fields(case.gold, pred)
                ph_report.cases.append(ModelCaseResult(text=case.text, gold=case.gold, pred=pred, latencies=lat, total_tokens=0, field_correct=fc))
            reports["phobert"] = ph_report
        except Exception as e:
            print("Warning: PhoBERT inference skipped due to error:", e)
            reports["phobert_error"] = ModelReport(name="PhoBERT_error")

    # 3) LLM Extractor - may raise if API key missing; catch and continue
    # 3) LLM Extractor
    if "llm" in methods:
        try:
            llm = LLMExtractor()
            llm_report = ModelReport(name="LLM")
            for case in cases:
                lat = []
                total_tokens = 0
                pred_obj = None
                for _ in range(REPEATS):
                    t0 = time.perf_counter()
                    # call low-level to get usage metadata
                    raw = llm._llm.invoke([{"role": "system", "content": llm_extractor__SYSTEM_PROMPT()}, {"role": "user", "content": case.text}])
                    lat.append(time.perf_counter() - t0)
                    # parse
                    pred_obj = llm._parse_response(raw.content)
                    usage = getattr(raw, "usage_metadata", None) or {}
                    total_tokens += usage.get("total_tokens") or 0
                pred = normalize_from_extracted(pred_obj)
                fc = compare_fields(case.gold, pred)
                llm_report.cases.append(ModelCaseResult(text=case.text, gold=case.gold, pred=pred, latencies=lat, total_tokens=total_tokens, field_correct=fc))
            reports["llm"] = llm_report
        except Exception as e:
            print("Warning: LLM inference skipped due to error:", e)
            reports["llm_error"] = ModelReport(name="LLM_error")

    return reports


def llm_extractor__has_system_prompt() -> bool:
    # helper to check the llm_extractor module for _SYSTEM_PROMPT
    try:
        from intent import llm_extractor

        return hasattr(llm_extractor, "_SYSTEM_PROMPT")
    except Exception:
        return False


def llm_extractor__SYSTEM_PROMPT() -> str:
    from intent import llm_extractor

    return getattr(llm_extractor, "_SYSTEM_PROMPT")


def pretty_print_summary(reports: Dict[str, ModelReport]) -> None:
    names = [k for k in ("rule", "phobert", "llm") if k in reports]
    print("=" * 60)
    print("FINAL BENCHMARK - 100 TEST CASES")
    print("=" * 60)
    print(f"{'Model':<18}{'Product':>10}{'Size':>8}{'Quantity':>12}{'Full Match':>12}")
    print("-" * 60)
    for key in ("rule", "phobert", "llm"):
        if key not in reports:
            continue
        r = reports[key]
        fa = r.field_accuracy
        print(f"{r.name:<18}{fa['product']*100:9.1f}%{fa['size']*100:8.1f}%{fa['quantity']*100:11.1f}%{r.full_match_accuracy*100:11.1f}%")
    print("\n")
    print(f"{'Model':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    print("-" * 60)
    for key in ("rule", "phobert", "llm"):
        if key not in reports:
            continue
        r = reports[key]
        # compute aggregate entity-level P/R/F1 across 3 entity types
        TP = FP = FN = 0
        for c in r.cases:
            for field, norm in (("product", normalize_product), ("size", normalize_size), ("quantity", normalize_quantity)):
                g = norm(c.gold.get(field))
                p = norm(c.pred.get(field))
                if g is not None and p == g:
                    TP += 1
                if p is not None and p != g:
                    FP += 1
                if g is not None and p != g:
                    FN += 1
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{r.name:<18}{prec*100:9.1f}%{rec*100:9.1f}%{f1*100:9.1f}%")
    print("\n")
    print(f"{'Model':<18}{'Avg Lat(ms)':>12}{'P50(ms)':>10}{'P95(ms)':>10}{'Cost':>12}")
    print("-" * 60)
    for key in ("rule", "phobert", "llm"):
        if key not in reports:
            continue
        r = reports[key]
        cost = "$0" if key in ("rule", "phobert") else (f"${r.total_cost_usd:.5f}" if r.total_tokens > 0 else "cost unavailable")
        print(f"{r.name:<18}{r.avg_latency*1000:12.1f}{r.p50_latency*1000:10.1f}{r.p95_latency*1000:10.1f}{cost:>12}")
    print("\n")


def save_results(path_json: str, path_md: str, cases: List[CaseRecord], reports: Dict[str, ModelReport]) -> None:
    # save detailed JSON
    out = {k: {"name": v.name, "cases": []} for k, v in reports.items()}
    for k, v in reports.items():
        for c in v.cases:
            out[k]["cases"].append({
                "text": c.text,
                "gold": c.gold,
                "pred": c.pred,
                "gold_normalized": {
                    "product": normalize_product(c.gold.get("product")),
                    "size": normalize_size(c.gold.get("size")),
                    "quantity": normalize_quantity(c.gold.get("quantity")),
                },
                "pred_normalized": {
                    "product": normalize_product(c.pred.get("product")),
                    "size": normalize_size(c.pred.get("size")),
                    "quantity": normalize_quantity(c.pred.get("quantity")),
                },
                "latencies_ms": [lt * 1000 for lt in c.latencies],
                "total_tokens": c.total_tokens,
                "field_correct": c.field_correct,
                "full_match": all(c.field_correct.values()),
                "errors": classify_error(c.gold, c.pred),
            })
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # write a simple markdown report
    lines: List[str] = []
    lines.append("# Benchmark Report\n")
    lines.append(f"Dataset: {len(cases)} cases")
    lines.append("\n## Summary Metrics\n")
    for k in ("rule", "phobert", "llm"):
        if k not in reports:
            lines.append(f"- {k}: not available")
            continue
        r = reports[k]
        lines.append(f"### {r.name}")
        lines.append(f"- Full extraction accuracy: {r.full_match_accuracy:.1%}")
        fa = r.field_accuracy
        lines.append(f"- Product accuracy: {fa['product']:.1%}")
        lines.append(f"- Size accuracy: {fa['size']:.1%}")
        lines.append(f"- Quantity accuracy: {fa['quantity']:.1%}")
        lines.append(f"- Avg latency (ms): {r.avg_latency*1000:.1f}")
        lines.append(f"- P50 (ms): {r.p50_latency*1000:.1f}")
        lines.append(f"- P95 (ms): {r.p95_latency*1000:.1f}")
        if k == "llm":
            lines.append(f"- Total tokens: {r.total_tokens}")
            lines.append(f"- Estimated cost: ${r.total_cost_usd:.5f}" if r.total_tokens else "- Cost: cost unavailable")
        lines.append("")

    lines.append("## Error Analysis\n")
    for k in ("rule", "phobert", "llm"):
        if k not in reports:
            continue
        lines.append(f"### {reports[k].name}")
        errs = []
        for c in reports[k].cases:
            e = classify_error(c.gold, c.pred)
            if e:
                errs.append((c.text, e, c.gold, c.pred))
        lines.append(f"- Total failing cases: {len(errs)}/{reports[k].n}")
        lines.append("")

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="ner_test_set_100.jsonl", help="Path to ner_test_set_100.jsonl (100 test cases, JSONL)")
    parser.add_argument("--out-json", default=os.path.join("intent", "benchmark_results.json"))
    parser.add_argument("--out-md", default=os.path.join("intent", "benchmark_report.md"))
    parser.add_argument(
        "--methods",
        default="rule,phobert,llm",
        help="Comma-separated methods to run: rule,phobert,llm (default: rule,phobert,llm - all 3, as required)",
    )
    args = parser.parse_args()

    try:
        cases = read_cases(args.cases)
    except Exception as e:
        print("Error reading cases:", e)
        return

    # measure
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    try:
        reports = measure_model_on_cases(cases, methods)
    except Exception as e:
        print("Error during measurement:", e)
        return

    pretty_print_summary(reports)

    save_results(args.out_json, args.out_md, cases, reports)


if __name__ == "__main__":
    main()
