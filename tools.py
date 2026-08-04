"""
tools.py
Các "tool" (function calling) mà LLM có thể gọi để thao tác đơn hàng.

Không dùng RAG/Qdrant ở đây - đây là dữ liệu có cấu trúc (menu.json) và
trạng thái phiên (giỏ hàng), nên xử lý bằng code thuần cho chính xác 100%,
thay vì để LLM tự suy đoán giá / cộng tiền (dễ sai số).

Nội dung:
0. Response helpers (error_response/success_response) - JSON thống nhất
1. Cache menu.json (đọc 1 lần, tự refresh nếu file thay đổi)
2. Suy luận size hợp lệ của từng món trực tiếp từ dữ liệu (không hardcode)
3. Product matching 4 tầng: exact -> normalized -> accent_insensitive ->
   fuzzy, có suggestions khi mơ hồ hoặc không tìm thấy
4. Giỏ hàng (shopping cart) + các tool: add_to_cart, view_cart,
   remove_from_cart, update_cart, clear_cart, checkout (đã tách
   subtotal/discount/shipping_fee/tax/total để dễ mở rộng)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    # Thư viện nhẹ, chuyên dụng để bỏ dấu Unicode (không chỉ tiếng Việt) -
    # dùng cho tầng matching "accent-insensitive" bên dưới.
    from unidecode import unidecode
except ImportError:  # pragma: no cover - vẫn chạy được nếu chưa `pip install`
    def unidecode(text: str) -> str:
        """Fallback KHÔNG cần cài thêm gói, dùng khi thiếu package `unidecode`
        (vd. môi trường chưa kịp cập nhật requirements.txt).

        Cách làm: 'đ/Đ' không tự tách dấu qua NFD nên xử lý riêng, các ký tự
        còn lại (à, ê, ố, ...) đều là "chữ cái gốc + dấu kết hợp" nên decompose
        bằng NFD rồi bỏ mọi ký tự thuộc nhóm dấu kết hợp (category 'Mn') là
        bỏ được dấu, không cần thư viện ngoài."""
        text = text.replace("đ", "d").replace("Đ", "D")
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_PATH = os.path.join(BASE_DIR, "data", "menu.json")

# 0. RESPONSE HELPERS (JSON có cấu trúc thống nhất)
# Trước đây mỗi tool tự viết lại json.dumps({"order_status": "error", ...})
# hoặc json.dumps({"order_status": "success", ...}) riêng lẻ ở nhiều nơi
# (add_to_cart, remove_from_cart, update_cart, checkout...). Gom về 2 hàm
# dùng chung giúp: (1) tránh lặp code, (2) đảm bảo MỌI response luôn có cùng
# khung {"order_status": ..., ...}, (3) sau này muốn đổi format (vd. thêm
# field "request_id") chỉ cần sửa 1 chỗ.
def _error_dict(error_type: str, message: str, **extra: object) -> dict:
    """Dict lỗi thống nhất: {"order_status": "error", "error_type", "message", **extra}."""
    payload = {"order_status": "error", "error_type": error_type, "message": message}
    payload.update(extra)
    return payload


def _success_dict(**fields: object) -> dict:
    """Dict thành công thống nhất: {"order_status": "success", **fields}."""
    return {"order_status": "success", **fields}


def error_response(error_type: str, message: str, **extra: object) -> str:
    """Chuỗi JSON lỗi - dùng làm return value trực tiếp cho các @tool.
    Cũng được chatbot.py tái sử dụng để mọi lỗi (kể cả lỗi tầng gọi tool,
    ngoài phạm vi 1 tool cụ thể) đều có chung 1 định dạng."""
    return json.dumps(_error_dict(error_type, message, **extra), ensure_ascii=False)


def success_response(**fields: object) -> str:
    """Chuỗi JSON thành công - dùng làm return value trực tiếp cho các @tool."""
    return json.dumps(_success_dict(**fields), ensure_ascii=False)

# 1. MENU LOADING + CACHE
# Trước đây mỗi lần gọi tool đều mở lại + json.load(menu.json) từ đầu - tốn
# I/O không cần thiết vì menu hiếm khi đổi trong lúc chatbot đang chạy.
# Cache theo mtime (thời điểm sửa file) để vẫn tự động nhận thay đổi mới
# nếu ai đó chạy lại ingest.py / sửa menu.json mà không cần restart chatbot.
_menu_cache: list[dict] | None = None
_menu_cache_mtime: float | None = None


def _load_menu() -> list[dict]:
    """Đọc menu.json, có cache. Chỉ đọc lại file khi mtime thay đổi."""
    global _menu_cache, _menu_cache_mtime
    mtime = os.path.getmtime(MENU_PATH)
    if _menu_cache is None or mtime != _menu_cache_mtime:
        with open(MENU_PATH, encoding="utf-8") as f:
            _menu_cache = json.load(f)
        _menu_cache_mtime = mtime
    return _menu_cache


def reload_menu() -> None:
    """Ép buộc đọc lại menu.json ở lần truy cập kế tiếp (bỏ qua cache).
    Hữu ích khi test hoặc khi biết chắc menu vừa được cập nhật."""
    global _menu_cache, _menu_cache_mtime
    _menu_cache = None
    _menu_cache_mtime = None

# 2. SIZE HỢP LỆ - SUY RA TỪ DỮ LIỆU, KHÔNG HARDCODE
# menu.json không có field "sizes" tường minh, nhưng mỗi món có các field dạng price_<size> (price_m, price_l, ...). Thay vì hardcode VALID_SIZES =
# {"M", "L"}, ta quét chính các field này -> nếu sau này thêm size mới (vd.
# price_xl cho size XL) thì chỉ cần sửa data, code không cần đổi.
_PRICE_FIELD_RE = re.compile(r"^price_(\w+)$")


def get_product_sizes(item: dict) -> dict[str, int | float]:
    """Trả về {TÊN_SIZE: giá} của 1 món, suy ra từ các field price_* có
    trong chính item đó. Vd: {"M": 25000, "L": 33000}."""
    sizes: dict[str, int | float] = {}
    for key, value in item.items():
        match = _PRICE_FIELD_RE.match(key)
        if match and isinstance(value, (int, float)):
            sizes[match.group(1).upper()] = value
    return sizes

# 3. PRODUCT MATCHING - 4 TẦNG
def _normalize(text: str) -> str:
    """Chuẩn hóa chuỗi: lower-case + gộp khoảng trắng thừa, để so khớp tên
    món không phân biệt hoa/thường hay cách gõ dư khoảng trắng.
    LƯU Ý: hàm này vẫn GIỮ DẤU tiếng Việt - dùng cho tầng exact/normalized
    (khách gõ có dấu). Muốn so khớp không dấu thì dùng thêm _fold()."""
    return " ".join(text.strip().lower().split())


def _fold(text: str) -> str:
    """Bỏ dấu tiếng Việt (và dấu Unicode nói chung) sau khi đã _normalize(),
    để so khớp kiểu "ca phe muoi" -> "Cà Phê Muối". Dùng `unidecode` (nhẹ,
    không cần model/dữ liệu ngoài) thay vì tự viết bảng ánh xạ ký tự thủ công
    - vừa gọn vừa xử lý đúng cho nhiều bảng mã hơn là tự chế."""
    return unidecode(text)


@dataclass
class ProductMatch:
    """Kết quả tìm món trong menu.

    status:
      - "exact"              : khớp chính xác tên (còn dấu, đã chuẩn hóa)
      - "normalized"         : khớp kiểu chứa nhau (substring, còn dấu)
      - "accent_insensitive" : khớp khi bỏ dấu (vd. "ca phe muoi" -> "Cà Phê Muối")
      - "fuzzy"               : khớp gần đúng (typo...) qua difflib trên bản KHÔNG dấu
      - "ambiguous"           : khớp nhiều hơn 1 món -> cần hỏi lại khách
      - "not_found"           : không tìm thấy món nào phù hợp
    """
    status: Literal[
        "exact", "normalized", "accent_insensitive", "fuzzy", "ambiguous", "not_found"
    ]
    product: dict | None = None
    suggestions: list[str] = field(default_factory=list)


def find_product(product_name: str) -> ProductMatch:
    """Tìm 1 món trong menu theo 4 tầng, dừng ở tầng đầu tiên cho kết quả
    rõ ràng (đúng 1 kết quả). Nếu 1 tầng cho ra nhiều kết quả, coi là
    "ambiguous" ngay (không đi tiếp xuống tầng sau) để tránh việc các tầng
    "lỏng" hơn (accent-insensitive, fuzzy) làm mờ thêm 1 kết quả vốn đã
    không rõ ràng ở tầng trước.

    Thứ tự vẫn giữ nguyên tinh thần cũ (exact -> normalized -> fuzzy), chỉ
    CHÈN THÊM 1 tầng "accent_insensitive" trước fuzzy để xử lý riêng trường
    hợp khách gõ không dấu (rất phổ biến khi chat tiếng Việt trên điện
    thoại) - trước đây trường hợp này phải trông chờ vào fuzzy match trên
    chuỗi CÓ dấu, độ chính xác thấp hơn hẳn vì difflib so khớp theo từng
    ký tự Unicode (à/ê/ố... là các ký tự khác hẳn a/e/o nên tỷ lệ khớp bị
    tính thấp một cách không cần thiết)."""
    menu = _load_menu()
    query = _normalize(product_name)
    if not query:
        return ProductMatch(status="not_found")

    # Map tên đã chuẩn hóa (còn dấu) -> item, dùng cho tầng 1 & 2.
    normalized_names = {_normalize(item["name"]): item for item in menu}

    # Tầng 1: khớp chính xác (còn dấu) 
    if query in normalized_names:
        return ProductMatch(status="exact", product=normalized_names[query])

    # Tầng 2: khớp chứa nhau, còn dấu (substring 2 chiều) 
    substring_hits = [
        item for norm_name, item in normalized_names.items()
        if query in norm_name or norm_name in query
    ]
    if len(substring_hits) == 1:
        return ProductMatch(status="normalized", product=substring_hits[0])
    if len(substring_hits) > 1:
        return ProductMatch(
            status="ambiguous",
            suggestions=[item["name"] for item in substring_hits[:5]],
        )

    # Map tên KHÔNG dấu -> item, dùng cho tầng 3 & 4. Tính 1 lần, dùng lại
    # cho cả accent-insensitive lẫn fuzzy để không fold() lặp lại.
    folded_names = {_fold(norm_name): item for norm_name, item in normalized_names.items()}
    query_folded = _fold(query)

    # Tầng 3: khớp không dấu (accent-insensitive), chính xác hoặc chứa nhau 
    if query_folded in folded_names:
        return ProductMatch(status="accent_insensitive", product=folded_names[query_folded])

    folded_substring_hits = [
        item for folded_name, item in folded_names.items()
        if query_folded in folded_name or folded_name in query_folded
    ]
    if len(folded_substring_hits) == 1:
        return ProductMatch(status="accent_insensitive", product=folded_substring_hits[0])
    if len(folded_substring_hits) > 1:
        return ProductMatch(
            status="ambiguous",
            suggestions=[item["name"] for item in folded_substring_hits[:5]],
        )

    # Tầng 4: fuzzy match trên chuỗi KHÔNG dấu (typo + thiếu dấu cùng lúc) 
    # Dùng difflib (thư viện chuẩn của Python, không cần cài thêm gói) -
    # đủ tốt cho menu ~40 món, không cần rapidfuzz/fuzzywuzzy.
    close = difflib.get_close_matches(query_folded, folded_names.keys(), n=5, cutoff=0.6)
    if len(close) == 1:
        return ProductMatch(status="fuzzy", product=folded_names[close[0]])
    if len(close) > 1:
        return ProductMatch(
            status="ambiguous",
            suggestions=[folded_names[name]["name"] for name in close],
        )

    return ProductMatch(status="not_found")


# 4. GIỎ HÀNG (SHOPPING CART)

@dataclass
class CartLine:
    """1 dòng trong giỏ hàng: 1 món + 1 size cụ thể."""
    product_name: str
    size: str
    unit_price: int | float
    quantity: int

    @property
    def line_total(self) -> int | float:
        return self.unit_price * self.quantity

    def to_dict(self) -> dict:
        return {
            "product_name": self.product_name,
            "size": self.size,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "line_total": self.line_total,
        }


# Giỏ hàng là state của PHIÊN CHAT hiện tại (giống conversation_history bên
# chatbot.py) - key = (tên món đã chuẩn hóa, size) để gộp số lượng nếu
# khách đặt trùng món + size ở nhiều lượt chat khác nhau.
_cart: dict[tuple[str, str], CartLine] = {}


def reset_cart() -> None:
    """Làm trống giỏ hàng. Gọi khi bắt đầu phiên mới (đồng bộ với
    chatbot.reset_history()) hoặc sau khi checkout() thành công."""
    global _cart
    _cart = {}


def _cart_key(product_name: str, size: str) -> tuple[str, str]:
    return (_normalize(product_name), size.strip().upper())


def _cart_summary() -> dict:
    """Snapshot giỏ hàng hiện tại: danh sách dòng + tổng tiền.
    Dùng chung cho mọi tool trả kết quả, để LLM luôn thấy trạng thái
    giỏ hàng mới nhất sau mỗi thao tác."""
    lines = [line.to_dict() for line in _cart.values()]
    subtotal = sum(line.line_total for line in _cart.values())
    return {
        "items": lines,
        "item_count": len(lines),
        "subtotal": subtotal,
        "currency": "VND",
    }


def _add_single_item(product_name: str, size: str, quantity: int) -> dict:
    """Validate + thêm 1 item vào giỏ hàng. Tách riêng khỏi tool add_to_cart
    để dùng lại logic tìm-món/validate size cho các tool khác nếu cần, và để
    add_to_cart có thể lặp qua nhiều item mà không lặp lại code validate."""
    if quantity <= 0:
        return _error_dict(
            "invalid_quantity",
            f"Số lượng '{quantity}' không hợp lệ, phải là số nguyên dương.",
            requested=product_name,
        )

    match = find_product(product_name)
    if match.status == "not_found":
        return _error_dict(
            "invalid_product",
            f"Không tìm thấy món '{product_name}' trong menu.",
            requested=product_name,
            suggestions=match.suggestions,
        )
    if match.status == "ambiguous":
        return _error_dict(
            "ambiguous_product",
            f"'{product_name}' khớp với nhiều món khác nhau, cần khách nói rõ hơn.",
            requested=product_name,
            suggestions=match.suggestions,
        )

    product = match.product
    sizes = get_product_sizes(product)
    size_norm = size.strip().upper()
    if size_norm not in sizes:
        return _error_dict(
            "invalid_size",
            f"Món '{product['name']}' không có size '{size}'.",
            requested=product_name,
            valid_sizes=sorted(sizes.keys()),
        )

    key = _cart_key(product["name"], size_norm)
    unit_price = sizes[size_norm]
    if key in _cart:
        _cart[key].quantity += quantity
    else:
        _cart[key] = CartLine(
            product_name=product["name"],
            size=size_norm,
            unit_price=unit_price,
            quantity=quantity,
        )

    return _success_dict(
        product_name=product["name"],
        size=size_norm,
        quantity_added=quantity,
        unit_price=unit_price,
        match_type=match.status,  # "exact"/"normalized"/"accent_insensitive"/"fuzzy" - để debug/log
    )


# --- Schema cho 1 dòng đặt món, dùng để add_to_cart nhận NHIỀU món/lần gọi ---
class OrderItem(BaseModel):
    """1 món khách muốn thêm vào giỏ hàng."""
    product_name: str = Field(description="Tên món, vd. 'Cà Phê Muối'.")
    size: str = Field(description="Size ly, vd. 'M' hoặc 'L' tuỳ món.")
    quantity: int = Field(description="Số lượng, số nguyên dương.", gt=0)


@tool
def add_to_cart(items: list[OrderItem]) -> str:
    """Thêm MỘT hoặc NHIỀU món vào giỏ hàng trong 1 lần gọi.

    CHỈ gọi khi khách rõ ràng muốn đặt/thêm món (vd. "cho tôi 1 ly cà phê
    muối size L và 2 bánh croissant"). Nếu khách đặt nhiều món trong 1 câu,
    hãy đưa TẤT CẢ vào cùng 1 lần gọi (list items), không gọi tool nhiều lần.

    Trả về JSON gồm:
    - results: kết quả xử lý từng item (success/error kèm lý do cụ thể)
    - cart: toàn bộ giỏ hàng hiện tại sau khi thêm (đã tính subtotal)
    """
    results = [_add_single_item(it.product_name, it.size, it.quantity) for it in items]
    return success_response(results=results, cart=_cart_summary())


@tool
def view_cart() -> str:
    """Xem giỏ hàng hiện tại. Dùng khi khách hỏi 'giỏ hàng có gì', 'tổng
    tiền bao nhiêu', 'tôi đặt những gì rồi'..."""
    return success_response(**_cart_summary())


def _find_cart_line_or_error(product_name: str, size: str) -> tuple[tuple[str, str] | None, str | None]:
    """Tìm 1 dòng trong giỏ theo (product_name, size) - cả 2 đều BẮT BUỘC.
    Trả về (key, None) nếu tìm thấy, hoặc (None, error_json) nếu không.

    Dùng cho remove_from_cart (xoá hẳn 1 dòng thì cần biết chính xác dòng
    nào). update_cart() dùng 1 helper khác (_locate_cart_line_for_update)
    vì size ở đó là TUỲ CHỌN - xem giải thích ở helper đó."""
    key = _cart_key(product_name, size)
    if key in _cart:
        return key, None

    match = find_product(product_name)
    error_json = error_response(
        "not_in_cart",
        f"Không tìm thấy '{product_name}' size '{size}' trong giỏ hàng.",
        suggestions=match.suggestions,
        cart=_cart_summary(),
    )
    return None, error_json


@tool
def remove_from_cart(product_name: str, size: str) -> str:
    """Xoá hẳn 1 món (theo tên + size) khỏi giỏ hàng."""
    key, error_json = _find_cart_line_or_error(product_name, size)
    if error_json is not None:
        return error_json

    removed = _cart.pop(key)
    return success_response(removed=removed.to_dict(), cart=_cart_summary())


def _locate_cart_line_for_update(product_name: str, current_size: str | None) -> tuple[tuple[str, str] | None, dict | None, str | None]:
    """Tìm 1 dòng trong giỏ để update_cart() sửa đổi.

    Khác với _find_cart_line_or_error() (dùng cho remove_from_cart, size là
    BẮT BUỘC), ở đây `current_size` là TUỲ CHỌN - vì khi khách nói "đổi
    latte thành 2 ly" mà giỏ chỉ có đúng 1 dòng Latte, không cần bắt khách
    nhắc lại size hiện tại.

    Trả về (key, product, error_json):
    - Tìm thấy đúng 1 dòng -> (key, product, None)
    - Không tìm thấy / mơ hồ (khách có nhiều size của cùng món trong giỏ mà
      không chỉ rõ đang muốn sửa size nào) -> (None, None, error_json)
    """
    match = find_product(product_name)
    if match.status in ("not_found", "ambiguous"):
        return None, None, error_response(
            "invalid_product",
            f"Không tìm thấy món '{product_name}' trong menu.",
            requested=product_name,
            suggestions=match.suggestions,
        )
    product = match.product
    norm_name = _normalize(product["name"])

    if current_size is not None:
        key = (norm_name, current_size.strip().upper())
        if key not in _cart:
            return None, None, error_response(
                "not_in_cart",
                f"Không tìm thấy '{product['name']}' size '{current_size}' trong giỏ hàng.",
                cart=_cart_summary(),
            )
        return key, product, None

    # Không chỉ rõ size hiện tại -> tự tìm trong các dòng đã có của món này.
    matching_keys = [key for key in _cart if key[0] == norm_name]
    if not matching_keys:
        return None, None, error_response(
            "not_in_cart",
            f"'{product['name']}' hiện không có trong giỏ hàng.",
            cart=_cart_summary(),
        )
    if len(matching_keys) > 1:
        return None, None, error_response(
            "ambiguous_cart_item",
            f"'{product['name']}' đang có nhiều size khác nhau trong giỏ hàng, "
            f"cần nói rõ đang muốn sửa size nào.",
            sizes_in_cart=sorted(key[1] for key in matching_keys),
            cart=_cart_summary(),
        )
    return matching_keys[0], product, None


@tool
def update_cart(
    product_name: str,
    current_size: str | None = None,
    new_size: str | None = None,
    new_quantity: int | None = None,
) -> str:
    """Cập nhật 1 món ĐÃ CÓ trong giỏ hàng: đổi size, đổi số lượng, hoặc cả hai.

    Dùng khi khách muốn SỬA một dòng đã đặt (không phải thêm món mới), ví dụ:
    - "đổi thành size M"                 -> chỉ new_size
    - "đổi thành size L"                 -> chỉ new_size
    - "đổi thành 3 ly"                   -> chỉ new_quantity
    - "đổi latte thành size M và 2 ly"   -> cả new_size lẫn new_quantity

    Args:
        product_name: Tên món cần sửa trong giỏ.
        current_size: Size HIỆN TẠI của dòng cần sửa. Có thể BỎ QUA nếu giỏ
            hàng chỉ có đúng 1 size của món này (trường hợp phổ biến nhất).
        new_size: Size MỚI muốn đổi sang. Bỏ qua nếu không đổi size.
        new_quantity: Số lượng MỚI. Đặt = 0 để xoá món khỏi giỏ. Bỏ qua nếu
            không đổi số lượng.

    Cần cung cấp ÍT NHẤT 1 trong 2: new_size hoặc new_quantity.
    """
    if new_size is None and new_quantity is None:
        return error_response(
            "no_changes_requested",
            "Cần cho biết size mới hoặc số lượng mới để cập nhật giỏ hàng.",
        )

    if new_quantity is not None and new_quantity < 0:
        return error_response(
            "invalid_quantity",
            f"Số lượng '{new_quantity}' không hợp lệ, phải là số nguyên >= 0.",
        )

    key, product, error_json = _locate_cart_line_for_update(product_name, current_size)
    if error_json is not None:
        return error_json

    line = _cart[key]

    # Đặt số lượng = 0 -> xoá thẳng dòng này, không cần xét đến new_size nữa.
    if new_quantity == 0:
        del _cart[key]
        return success_response(
            message=f"Đã xoá '{product['name']}' khỏi giỏ hàng vì số lượng = 0.",
            cart=_cart_summary(),
        )

    target_size = line.size
    if new_size is not None:
        sizes = get_product_sizes(product)
        new_size_norm = new_size.strip().upper()
        if new_size_norm not in sizes:
            return error_response(
                "invalid_size",
                f"Món '{product['name']}' không có size '{new_size}'.",
                valid_sizes=sorted(sizes.keys()),
            )
        target_size = new_size_norm

    target_quantity = new_quantity if new_quantity is not None else line.quantity
    target_key = _cart_key(product["name"], target_size)
    target_unit_price = get_product_sizes(product)[target_size]

    if target_key == key:
        # Không đổi size (hoặc đổi "size mới" trùng size cũ) -> sửa tại chỗ.
        _cart[key].quantity = target_quantity
    else:
        # Đổi sang size khác: xoá dòng cũ; nếu giỏ đã có sẵn 1 dòng khác cùng
        # size mới đó thì GỘP số lượng lại (giống cách add_to_cart gộp khi
        # trùng món+size), thay vì tạo 2 dòng trùng nhau trong giỏ.
        del _cart[key]
        if target_key in _cart:
            _cart[target_key].quantity += target_quantity
        else:
            _cart[target_key] = CartLine(
                product_name=product["name"],
                size=target_size,
                unit_price=target_unit_price,
                quantity=target_quantity,
            )

    return success_response(
        product_name=product["name"],
        size=target_size,
        quantity=target_quantity,
        unit_price=target_unit_price,
        cart=_cart_summary(),
    )


@tool
def clear_cart() -> str:
    """Xoá TOÀN BỘ giỏ hàng (dùng khi khách muốn huỷ hết, đặt lại từ đầu)."""
    reset_cart()
    return success_response(message="Đã xoá toàn bộ giỏ hàng.", cart=_cart_summary())


# Các thành phần cấu thành TOTAL, tách riêng từng hàm để checkout() dễ
# mở rộng về sau (vd. cắm logic khuyến mãi từ promotions.md, tính ship theo
# khoảng cách, áp % VAT...) mà không cần sửa lại phần tính tổng bên dưới.
# Hiện tại tất cả đều trả 0 - CHƯA có tính năng khuyến mãi/ship/thuế thật.
def _compute_discount(cart_summary: dict) -> int | float:
    """Số tiền được giảm giá. Hiện = 0 (chưa tích hợp khuyến mãi tự động)."""
    return 0


def _compute_shipping_fee(cart_summary: dict) -> int | float:
    """Phí giao hàng. Hiện = 0 (giả định khách uống tại quán/lấy tại quầy,
    chưa hỗ trợ đặt giao hàng qua chatbot)."""
    return 0


def _compute_tax(cart_summary: dict) -> int | float:
    """Thuế (vd. VAT). Hiện = 0 - giá trong menu.json được coi là giá bán
    cuối cùng, chưa tách riêng phần thuế."""
    return 0


@tool
def checkout() -> str:
    """Chốt đơn hàng từ giỏ hàng hiện tại.

    CHỈ gọi khi khách đã XÁC NHẬN muốn chốt đơn (vd. "chốt đơn giúp tôi",
    "vậy là xong rồi", "thanh toán"). KHÔNG tự ý checkout khi khách chỉ mới
    thêm món mà chưa xác nhận xong. Sau khi checkout thành công, giỏ hàng
    sẽ được làm trống để sẵn sàng cho đơn tiếp theo.
    """
    if not _cart:
        return error_response("empty_cart", "Giỏ hàng đang trống, chưa có món nào để chốt đơn.")

    summary = _cart_summary()
    subtotal = summary["subtotal"]
    discount = _compute_discount(summary)
    shipping_fee = _compute_shipping_fee(summary)
    tax = _compute_tax(summary)
    total = subtotal - discount + shipping_fee + tax

    receipt = success_response(
        items=summary["items"],
        item_count=summary["item_count"],
        subtotal=subtotal,
        discount=discount,
        shipping_fee=shipping_fee,
        tax=tax,
        total=total,
        currency="VND",
        checked_out_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    reset_cart()
    return receipt


# Danh sách tool để import gọn vào chatbot.py
ALL_TOOLS = [add_to_cart, view_cart, remove_from_cart, update_cart, clear_cart, checkout]


if __name__ == "__main__":
    # Test nhanh luồng: thêm nhiều món (kể cả gõ không dấu) -> xem giỏ ->
    # đổi size -> đổi số lượng -> đổi cả 2 cùng lúc -> checkout
    print(find_product("ca phe muoi"))  # kỳ vọng status="accent_insensitive"
    print(add_to_cart.invoke({"items": [
        {"product_name": "ca phe muoi", "size": "l", "quantity": 2},   # không dấu hoàn toàn
        {"product_name": "Bánh Croissant Bơ Tỏi", "size": "M", "quantity": 1},
        {"product_name": "Món Không Tồn Tại", "size": "M", "quantity": 1},
    ]}))
    print(view_cart.invoke({}))
    # "đổi thành 3 ly" - chỉ có 1 size Cà Phê Muối trong giỏ nên không cần current_size
    print(update_cart.invoke({"product_name": "Cà Phê Muối", "new_quantity": 3}))
    # "đổi thành size M" - đổi size, giữ nguyên số lượng
    print(update_cart.invoke({"product_name": "Cà Phê Muối", "new_size": "M"}))
    print(checkout.invoke({}))
