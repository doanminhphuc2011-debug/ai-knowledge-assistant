"""Chatbot orchestration.
Intent routing is LLM-driven. Entity extraction is generic. Tool readiness is
validated against each tool's schema instead of a hard-coded entity list.
"""
from __future__ import annotations
import logging
import os
import warnings
# Hugging Face / Transformers: ẩn progress bar và advisory warning
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", module=r"huggingface_hub\..*")
warnings.filterwarnings("ignore", message=r".*multilingual-e5-large now uses mean pooling.*", category=UserWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_core.messages import AIMessage, ToolMessage
from context_management import ContextBlock
from context_runtime import get_context_manager
from intent.extractor_factory import get_extractor
from intent.product_validation_strategy import (ProductValidationStrategy, get_product_validation_strategy)
from llm import llm
from quota_management import QuotaExceededError
from rag import retrieve_context
from tool_argument_builder import ToolArgumentBuilder
from tool_executor import _execute_tool_call, generate_with_tools
from tools.cart import reset_cart
# Ẩn log không cần thiết của Hugging Face khi demo
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
_context_manager = get_context_manager()
_tool_argument_builder = ToolArgumentBuilder()

_llm_intent_extractor = None
_phobert_ner_extractor = None
_llm_intent_init_failed = False
_phobert_init_failed = False
_product_validation_strategy: ProductValidationStrategy = get_product_validation_strategy()

def _get_llm_intent_extractor():
    global _llm_intent_extractor, _llm_intent_init_failed
    if _llm_intent_extractor is not None:
        return _llm_intent_extractor
    if _llm_intent_init_failed:
        return None
    try:
        _llm_intent_extractor = get_extractor("llm")
        return _llm_intent_extractor
    except Exception:
        _llm_intent_init_failed = True
        logger.exception("[ROUTER] Không khởi tạo được LLM Intent Classifier")
        return None

def _get_phobert_ner_extractor():
    global _phobert_ner_extractor, _phobert_init_failed
    if _phobert_ner_extractor is not None:
        return _phobert_ner_extractor
    if _phobert_init_failed:
        return None
    try:
        _phobert_ner_extractor = get_extractor("phobert")
        return _phobert_ner_extractor
    except Exception:
        _phobert_init_failed = True
        logger.exception("[ROUTER] Không khởi tạo được PhoBERT NER")
        return None

def _run_intent_classification(question: str):
    extractor = _get_llm_intent_extractor()
    if extractor is None:
        return None
    try:
        return extractor.extract(question)
    except QuotaExceededError:
        raise
    except Exception:
        logger.exception("[INTENT] LLM classification failed")
        return None

def _run_ner(question: str):
    extractor = _get_phobert_ner_extractor()
    if extractor is None:
        return None
    try:
        result = extractor.extract(question)
    except Exception:
        logger.exception("[NER] PhoBERT extraction failed")
        return None
    logger.debug("[NER] entities=%s", result["entities"])
    return result

def _validate_product(llm_product: str | None, ner_product: str | None):
    return _product_validation_strategy.validate(llm_product=llm_product, ner_product=ner_product)

def _inject_tool_result(messages, tool_name: str, tool_args: dict, tool_result: str):
    tool_call = {"name": tool_name, "args": tool_args, "id": "router-call-1"}
    logger.debug("[TOOL] %s(%s)", tool_name, tool_args)
    return messages + [
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content=tool_result, tool_call_id="router-call-1"),
    ]

def _generate_with_rag(question: str, tool_name: str | None = None, tool_args: dict | None = None, session_id: str | None = None) -> str:
    retrieved_context = retrieve_context(question)
    context_blocks = (
        [ContextBlock(name="retrieval", content=retrieved_context)]
        if retrieved_context
        else []
    )

    assembly = _context_manager.prepare(user_input=question, session_id=session_id, context_blocks=context_blocks)
    messages_for_llm = assembly.messages

    logger.debug(
        "[CONTEXT] session=%s input_tokens~%s history_used=%s history_dropped=%s external_tokens~%s",
        assembly.session_id,
        assembly.estimated_input_tokens,
        assembly.history_messages_used,
        assembly.history_messages_dropped,
        assembly.external_context_tokens,
    )

    if tool_name is not None and tool_args is not None:
        tool_call = {"name": tool_name, "args": tool_args, "id": "router-call-1"}
        tool_result = _execute_tool_call(tool_call)
        messages_for_llm = _inject_tool_result(
            messages_for_llm, tool_name, tool_args, tool_result
        )

    answer = generate_with_tools(llm, messages_for_llm)
    _context_manager.record_turn(user_input=question, assistant_output=answer, session_id=assembly.session_id)
    return answer

def _ask_impl(question: str, session_id: str | None = None) -> str:
    intent_result = _run_intent_classification(question)
    if intent_result is None:
        return _generate_with_rag(question, session_id=session_id)

    intent = intent_result["intent"]

    # Information intents do not correspond to a registered business tool.
    if not _tool_argument_builder.has_tool(intent):
        return _generate_with_rag(question, session_id=session_id)

    # Parameterless tools (e.g. view_cart/checkout) can be called immediately.
    direct = _tool_argument_builder.build(intent, {})
    if direct.ready:
        return _generate_with_rag(question, tool_name=direct.tool_name, tool_args=direct.arguments, session_id=session_id)

    ner_result = _run_ner(question)
    if ner_result is None:
        return _generate_with_rag(question, session_id=session_id)

    ner_entities = ner_result["entities"]
    llm_entities = intent_result["entities"]

    # Product consistency is checked only when the NER result actually contains
    # a product. Tools without product slots are unaffected.
    ner_product = ner_entities.get("product_name")
    llm_product = llm_entities.get("product_name")
    if isinstance(ner_product, str):
        validation = _validate_product(llm_product=llm_product if isinstance(llm_product, str) else None, ner_product=ner_product)
        if not validation.matched:
            logger.debug("[ROUTER] PRODUCT MISMATCH | LLM=%r | PhoBERT=%r", validation.llm_product, validation.ner_product)
            return _generate_with_rag(question, session_id=session_id)

    build = _tool_argument_builder.build(intent, ner_entities)
    if not build.ready:
        logger.debug("[ROUTER] Missing required tool slots: %s", build.missing_required)
        return _generate_with_rag(question, session_id=session_id)

    return _generate_with_rag(question, tool_name=build.tool_name, tool_args=build.arguments, session_id=session_id)

def ask(question: str, session_id: str | None = None) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question phải là chuỗi không rỗng")
    question = question.strip()

    try:
        return _ask_impl(question, session_id=session_id)
    except QuotaExceededError as exc:
        logger.warning(
            "[QUOTA] resource=%s used=%s limit=%s retry_after=%ss",
            exc.resource,
            exc.used,
            exc.limit,
            exc.retry_after_seconds,
        )
        return ("Hệ thống đang đạt giới hạn sử dụng tạm thời. " f"Vui lòng thử lại sau khoảng {exc.retry_after_seconds} giây.")

def reset_history(session_id: str | None = None) -> None:
    _context_manager.clear_session(session_id)
    reset_cart()

chat = ask

if __name__ == "__main__":
    print("☕ Ori - Trợ lý quán cà phê DMP")
    print("Nhập 'exit' để thoát.")
    while True:
        user_input = input("👤 Bạn: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        print(f"Ori: {ask(user_input)}\n")
