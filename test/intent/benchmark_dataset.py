"""
intent/benchmark_dataset.py
80 test case cho benchmark LLMExtractor vs NERExtractor, chia 7 nhóm.

NGUYÊN TẮC XÂY DỰNG `expected` (ĐỌC TRƯỚC KHI SỬA FILE NÀY):
1. `expected` là NGHĨA THẬT của câu nói (ground truth), xác định TRƯỚC và
   ĐỘC LẬP với việc chạy thử LLMExtractor/NERExtractor - KHÔNG được suy
   ngược từ output của bất kỳ extractor nào (yêu cầu bắt buộc từ đề bài).
2. product_name/size LUÔN đối chiếu với data/menu.json thật (xem
   `_ALL_PRODUCT_NAMES` cuối file - assert tại import time để phát hiện
   ngay nếu 1 case nào đó lỡ dùng tên món không tồn tại).
3. Field nào câu nói KHÔNG nêu rõ -> expected = None. KHÔNG tự suy diễn
   "nói ly thì chắc là 1 ly" - chỉ gán quantity khi có SỐ TƯỜNG MINH (chữ
   số hoặc số đếm bằng chữ) xuất hiện trong câu.
4. QUY ƯỚC RIÊNG (quyết định TRƯỚC khi benchmark, không phải suy từ kết
   quả extractor): nói TÊN MÓN TRẦN không kèm động từ (vd. "cà phê sữa đá
   size L") trong ngữ cảnh voice chatbot ĐẶT ĐỒ UỐNG được hiểu là ý định
   "add_to_cart" - đây là quy ước NGHIỆP VỤ hợp lý của domain (khách nói
   tên món với bot gọi món gần như luôn có nghĩa là muốn gọi món đó),
   không phải quy tắc suy ra a-posteriori. Áp dụng nhất quán cho MỌI case
   dạng này trong cả 7 nhóm, không riêng nhóm nào.
5. size chỉ nhận "M" hoặc "L" (menu thật CHỈ có 2 size này cho toàn bộ 40
   món - xác nhận bằng cách quét data/menu.json, không giả định). Các từ
   mô tả "nhỏ"/"vừa" -> M, "lớn"/"to" -> L là quy ước ĐÃ CÓ SẴN trong
   NERExtractor (_SIZE_WORDS_RAW) - dataset dùng LẠI đúng quy ước đó để
   nhất quán trên toàn hệ thống, không tự đặt quy ước khác.

NHÓM F (sai chính tả/lỗi STT) CỐ Ý gồm cả 2 loại lỗi khác nhau về bản
chất, để benchmark phản ánh đúng năng lực thật thay vì làm dataset dễ:
  (a) Lỗi CHỈ MẤT DẤU/SAI DẤU THANH (vd. "sửa" thay vì "sữa", "câu" thay
      vì "cầu") - những lỗi này KHÔNG ảnh hưởng NER vì NER fold bỏ dấu
      trước khi so khớp (unidecode xóa cả dấu thanh điệu) - product_name
      NGHĨA THẬT không đổi, vẫn suy ra được đúng món.
  (b) Lỗi chính tả THẬT (thêm/thiếu/sai ký tự khác dấu, vd. "capuchino"
      thay vì "cappuccino", "late" thay vì "latte") - đây LÀ TRƯỜNG HỢP
      NER NHIỀU KHẢ NĂNG SẼ SAI (so khớp substring chính xác, không có
      fuzzy matching) - expected vẫn phải là NGHĨA THẬT (món đúng), để
      benchmark ghi nhận trung thực NER sai ở đây, không né tránh.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from intent.extractor_base import ExtractedEntities


@dataclass
class TestCase:
    text: str
    expected: ExtractedEntities
    category: str  # "basic" | "natural" | "missing_info" | "size" | "quantity" | "spelling_stt" | "non_order"


def _E(
    intent: str,
    product_name: str | None = None,
    size: str | None = None,
    quantity: int | None = None,
) -> ExtractedEntities:
    return ExtractedEntities(intent=intent, product_name=product_name, size=size, quantity=quantity)

# A. ĐẶT MÓN CƠ BẢN (15 case)
_GROUP_A: list[TestCase] = [
    TestCase("cho tôi cà phê sữa đá size L", _E("add_to_cart", "Cà Phê Sữa Đá", "L"), "basic"),
    TestCase("cho tôi 2 ly bạc xỉu", _E("add_to_cart", "Bạc Xỉu", None, 2), "basic"),
    TestCase("cho tôi một latte size M", _E("add_to_cart", "Latte", "M", 1), "basic"),
    TestCase("thêm một cà phê muối", _E("add_to_cart", "Cà Phê Muối", None, 1), "basic"),
    TestCase("cho tôi cappuccino size L", _E("add_to_cart", "Cappuccino", "L"), "basic"),
    TestCase("cho tôi 3 ly trà đào sả cam size M", _E("add_to_cart", "Trà Đào Sả Cam", "M", 3), "basic"),
    TestCase("cho tôi americano", _E("add_to_cart", "Americano"), "basic"),
    TestCase("cho tôi 1 ly matcha latte uji size L", _E("add_to_cart", "Matcha Latte Uji", "L", 1), "basic"),
    TestCase("cho tôi mocha cacao size M", _E("add_to_cart", "Mocha Cacao", "M"), "basic"),
    TestCase("cho tôi 2 ly trà vải nhài", _E("add_to_cart", "Trà Vải Nhài", None, 2), "basic"),
    TestCase("cho tôi caramel macchiato size L", _E("add_to_cart", "Caramel Macchiato", "L"), "basic"),
    TestCase("cho tôi bánh tiramisu truyền thống", _E("add_to_cart", "Bánh Tiramisu Truyền Thống"), "basic"),
    TestCase("cho tôi 1 ly cold brew truyền thống size M", _E("add_to_cart", "Cold Brew Truyền Thống", "M", 1), "basic"),
    TestCase("cho tôi sinh tố bơ dừa size L", _E("add_to_cart", "Sinh Tố Bơ Dừa", "L"), "basic"),
    TestCase("cho tôi 2 bánh croissant bơ tỏi", _E("add_to_cart", "Bánh Croissant Bơ Tỏi", None, 2), "basic"),
]

# B. CÁCH NÓI TỰ NHIÊN (10 case) - có từ đệm, xưng hô, tiểu từ cuối câu
_GROUP_B: list[TestCase] = [
    TestCase("anh cho em một ly cà phê sữa đá size lớn", _E("add_to_cart", "Cà Phê Sữa Đá", "L", 1), "natural"),
    TestCase("cho mình 2 bạc xỉu nha", _E("add_to_cart", "Bạc Xỉu", None, 2), "natural"),
    TestCase("em muốn gọi một ly latte", _E("add_to_cart", "Latte", None, 1), "natural"),
    TestCase("mình lấy cà phê muối nhé", _E("add_to_cart", "Cà Phê Muối"), "natural"),
    TestCase("chị ơi cho em ly trà đào sả cam size to", _E("add_to_cart", "Trà Đào Sả Cam", "L"), "natural"),
    TestCase("cho mình một trà sữa ô long nướng size vừa", _E("add_to_cart", "Trà Sữa Ô Long Nướng", "M", 1), "natural"),
    TestCase("anh lấy giúp em ly cappuccino", _E("add_to_cart", "Cappuccino"), "natural"),
    TestCase("em order một ly mocha cacao size lớn", _E("add_to_cart", "Mocha Cacao", "L", 1), "natural"),
    TestCase("cho chị 2 ly trà mãng cầu tươi nha", _E("add_to_cart", "Trà Mãng Cầu Tươi", None, 2), "natural"),
    TestCase("mình muốn một phần bánh cheesecake chanh dây", _E("add_to_cart", "Bánh Cheesecake Chanh Dây", None, 1), "natural"),
]

# C. THIẾU THÔNG TIN (10 case) - field không nói ra PHẢI là None
_GROUP_C: list[TestCase] = [
    TestCase("cho tôi cà phê sữa đá", _E("add_to_cart", "Cà Phê Sữa Đá"), "missing_info"),
    TestCase("cho tôi một ly latte", _E("add_to_cart", "Latte", None, 1), "missing_info"),
    # Tên món trần + size, không động từ - áp quy ước #4 ở docstring đầu file.
    TestCase("cà phê sữa đá size L", _E("add_to_cart", "Cà Phê Sữa Đá", "L"), "missing_info"),
    TestCase("2 ly bạc xỉu", _E("add_to_cart", "Bạc Xỉu", None, 2), "missing_info"),
    TestCase("cho tôi trà sữa thái xanh", _E("add_to_cart", "Trà Sữa Thái Xanh"), "missing_info"),
    TestCase("một ly americano", _E("add_to_cart", "Americano", None, 1), "missing_info"),
    TestCase("cho tôi caramel coffee freeze", _E("add_to_cart", "Caramel Coffee Freeze"), "missing_info"),
    TestCase("trà đào sả cam", _E("add_to_cart", "Trà Đào Sả Cam"), "missing_info"),
    TestCase("cho tôi 3 ly matcha đá xay đậu đỏ", _E("add_to_cart", "Matcha Đá Xay Đậu Đỏ", None, 3), "missing_info"),
    TestCase("sinh tố mango passion size L", _E("add_to_cart", "Sinh Tố Mango Passion", "L"), "missing_info"),
]

# D. CÁCH DIỄN ĐẠT SIZE (10 case) - chỉ dùng size M/L thật có trong menu
_GROUP_D: list[TestCase] = [
    TestCase("cho tôi cà phê đen đá size L", _E("add_to_cart", "Cà Phê Đen Đá", "L"), "size"),
    TestCase("cho tôi cà phê đen đá size lớn", _E("add_to_cart", "Cà Phê Đen Đá", "L"), "size"),
    TestCase("cho tôi cà phê đen đá ly lớn", _E("add_to_cart", "Cà Phê Đen Đá", "L"), "size"),
    TestCase("size M cà phê sữa đá", _E("add_to_cart", "Cà Phê Sữa Đá", "M"), "size"),
    TestCase("cho tôi cà phê sữa đá size vừa", _E("add_to_cart", "Cà Phê Sữa Đá", "M"), "size"),
    # Menu CHỈ có M/L (không có size nhỏ hơn M) - áp quy ước sẵn có của
    # NERExtractor: "nhỏ" -> M (size nhỏ nhất thực tế có bán).
    TestCase("cho tôi cà phê sữa đá size nhỏ", _E("add_to_cart", "Cà Phê Sữa Đá", "M"), "size"),
    TestCase("cho tôi bạc xỉu ly nhỏ", _E("add_to_cart", "Bạc Xỉu", "M"), "size"),
    TestCase("cho tôi trà vải nhài size to", _E("add_to_cart", "Trà Vải Nhài", "L"), "size"),
    TestCase("cho tôi latte cỡ lớn", _E("add_to_cart", "Latte", "L"), "size"),
    TestCase("cho tôi cappuccino M", _E("add_to_cart", "Cappuccino", "M"), "size"),
]

# E. SỐ LƯỢNG (10 case) - cả số dạng chữ và dạng số
_GROUP_E: list[TestCase] = [
    TestCase("1 ly latte", _E("add_to_cart", "Latte", None, 1), "quantity"),
    TestCase("2 ly latte", _E("add_to_cart", "Latte", None, 2), "quantity"),
    TestCase("ba ly bạc xỉu", _E("add_to_cart", "Bạc Xỉu", None, 3), "quantity"),
    TestCase("một ly cà phê muối", _E("add_to_cart", "Cà Phê Muối", None, 1), "quantity"),
    # Có số lượng nhưng KHÔNG nói rõ món - product_name PHẢI là None.
    TestCase("cho tôi năm ly", _E("add_to_cart", None, None, 5), "quantity"),
    TestCase("cho tôi 4 ly trà tắc xí muội", _E("add_to_cart", "Trà Tắc Xí Muội", None, 4), "quantity"),
    TestCase("cho tôi sáu ly trà chanh giã tay quảng đông", _E("add_to_cart", "Trà Chanh Giã Tay Quảng Đông", None, 6), "quantity"),
    TestCase("cho tôi 10 ly cà phê đen đá", _E("add_to_cart", "Cà Phê Đen Đá", None, 10), "quantity"),
    TestCase("cho tôi mười ly bạc xỉu", _E("add_to_cart", "Bạc Xỉu", None, 10), "quantity"),
    TestCase("cho tôi 7 ly trà sữa phô mai tươi", _E("add_to_cart", "Trà Sữa Phô Mai Tươi", None, 7), "quantity"),
]

# F. SAI CHÍNH TẢ / LỖI STT (15 case) - xem giải thích 2 LOẠI lỗi ở docstring đầu file
_GROUP_F: list[TestCase] = [
    # (a) chỉ mất dấu/sai dấu thanh - KHÔNG đổi nghĩa, fold vẫn khớp đúng
    TestCase("cho tôi caphe sua da size l", _E("add_to_cart", "Cà Phê Sữa Đá", "L"), "spelling_stt"),
    TestCase("cho tôi cafe sua da size lon", _E("add_to_cart", "Cà Phê Sữa Đá", "L"), "spelling_stt"),
    TestCase("cho tôi bac xiu", _E("add_to_cart", "Bạc Xỉu"), "spelling_stt"),
    TestCase("cho tôi cà phê sửa đá size L", _E("add_to_cart", "Cà Phê Sữa Đá", "L"), "spelling_stt"),
    TestCase("cho tôi bạc xỉu size nớn", _E("add_to_cart", "Bạc Xỉu", "L"), "spelling_stt"),
    TestCase("cho tôi cafe muoi", _E("add_to_cart", "Cà Phê Muối"), "spelling_stt"),
    TestCase("cho tôi tra dao sa cam size L", _E("add_to_cart", "Trà Đào Sả Cam", "L"), "spelling_stt"),
    TestCase("cho tôi trà mãng câu tươi", _E("add_to_cart", "Trà Mãng Cầu Tươi"), "spelling_stt"),
    TestCase("cho tôi trà sựa dmp truyền thống", _E("add_to_cart", "Trà Sữa DMP Truyền Thống"), "spelling_stt"),
    TestCase("cho tôi trà xoai macchiato", _E("add_to_cart", "Trà Xoài Macchiato"), "spelling_stt"),
    # (b) lỗi chính tả THẬT (thêm/thiếu/sai ký tự) - NER nhiều khả năng SAI ở đây, expected VẪN LÀ nghĩa thật
    TestCase("cho toi mot ly capuchino", _E("add_to_cart", "Cappuccino", None, 1), "spelling_stt"),
    TestCase("cho tôi 2 ly balc xiu", _E("add_to_cart", "Bạc Xỉu", None, 2), "spelling_stt"),
    TestCase("cho tôi cold brew cam xa", _E("add_to_cart", "Cold Brew Cam Sả"), "spelling_stt"),
    TestCase("cho tôi mocha caco", _E("add_to_cart", "Mocha Cacao"), "spelling_stt"),
    TestCase("cho tôi 1 ly late", _E("add_to_cart", "Latte", None, 1), "spelling_stt"),
]

# G. KHÔNG PHẢI ĐẶT MÓN (10 case) - kiểm tra không bị nhận nhầm add_to_cart
_GROUP_G: list[TestCase] = [
    TestCase("xin chào", _E("unknown"), "non_order"),
    TestCase("xem menu", _E("unknown"), "non_order"),
    TestCase("giá cà phê bao nhiêu", _E("unknown"), "non_order"),
    TestCase("quán mở cửa mấy giờ", _E("unknown"), "non_order"),
    TestCase("có khuyến mãi không", _E("unknown"), "non_order"),
    TestCase("xem giỏ hàng", _E("view_cart"), "non_order"),
    TestCase("thanh toán", _E("checkout"), "non_order"),
    TestCase("quán có phòng riêng không", _E("unknown"), "non_order"),
    TestCase("wifi quán là gì vậy", _E("unknown"), "non_order"),
    TestCase("cảm ơn nha", _E("unknown"), "non_order"),
]

ALL_TEST_CASES: list[TestCase] = (
    _GROUP_A + _GROUP_B + _GROUP_C + _GROUP_D + _GROUP_E + _GROUP_F + _GROUP_G
)

CATEGORY_LABELS: dict[str, str] = {
    "basic": "Basic (đặt món cơ bản)",
    "natural": "Natural (cách nói tự nhiên)",
    "missing_info": "Missing information (thiếu thông tin)",
    "size": "Size (cách diễn đạt size)",
    "quantity": "Quantity (số lượng)",
    "spelling_stt": "Spelling/STT (sai chính tả / lỗi STT)",
    "non_order": "Non-order (không phải đặt món)",
}


def _validate_dataset() -> None:
    """Kiểm tra dataset tự nhất quán NGAY LÚC IMPORT MODULE - phát hiện
    sớm nếu 1 case nào đó lỡ dùng tên món/size không tồn tại trong menu
    thật, thay vì để benchmark chạy xong mới phát hiện dataset sai."""
    with open("data/menu.json", encoding="utf-8") as f:
        menu = json.load(f)
    valid_names = {item["name"] for item in menu}

    assert len(ALL_TEST_CASES) == 80, f"Kỳ vọng 80 test case, hiện có {len(ALL_TEST_CASES)}"

    counts: dict[str, int] = {}
    for case in ALL_TEST_CASES:
        counts[case.category] = counts.get(case.category, 0) + 1
        name = case.expected.get("product_name")
        if name is not None and name not in valid_names:
            raise AssertionError(
                f"Test case {case.text!r} dùng product_name={name!r} "
                f"KHÔNG tồn tại trong data/menu.json"
            )
        size = case.expected.get("size")
        if size is not None and size not in ("M", "L"):
            raise AssertionError(f"Test case {case.text!r} dùng size={size!r} không hợp lệ (chỉ M/L)")

    expected_counts = {
        "basic": 15, "natural": 10, "missing_info": 10, "size": 10,
        "quantity": 10, "spelling_stt": 15, "non_order": 10,
    }
    assert counts == expected_counts, f"Số lượng case/nhóm không khớp: {counts} != {expected_counts}"


_validate_dataset()
