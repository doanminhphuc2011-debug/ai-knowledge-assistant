"""
voice_main.py
Entry point tối giản để chạy Voice Chat THẬT (không mock, không test) -
dùng đúng các module đã hoàn thiện ở các Phase trước (voice/microphone.py,
voice/stt.py, voice/text_normalizer.py, voice/tts.py, voice/speaker.py,
voice/voice_chat.py).

Không menu, không CLI parser (argparse), không tuỳ chọn cấu hình qua
tham số dòng lệnh - chỉ có đúng 1 vòng lặp: lắng nghe 1 lượt, xử lý, lặp
lại, dừng khi người dùng nhấn Ctrl+C (KeyboardInterrupt). Toàn bộ cấu
hình (model, ngôn ngữ, sample rate, device...) đến từ VoiceConfig (đọc
.env qua voice/config.py) - file này không hardcode bất kỳ giá trị cấu
hình nghiệp vụ nào ngoài thời lượng ghi âm mỗi lượt.

LƯU Ý VỀ THAM SỐ `duration`: `VoiceChat.listen_once()` (voice/voice_chat.py,
Phase 6) có chữ ký `listen_once(self, duration: float) -> str` - tên
tham số là `duration`, KHÔNG PHẢI `duration_seconds`. Gọi bằng keyword
`duration_seconds=...` sẽ ném `TypeError` ngay lập tức vì không khớp
tên tham số thực tế của method đã tồn tại. voice_main.py vì vậy gọi
đúng `duration=RECORDING_DURATION_SECONDS` - đây là điểm khác biệt DUY
NHẤT so với đoạn code mẫu trong yêu cầu Phase 7B (đoạn mẫu dùng tên
`duration_seconds`). Không tự ý sửa voice/voice_chat.py để đổi tên tham
số cho khớp mẫu - xem giải thích chi tiết trong câu trả lời kèm theo.

Bất kỳ lỗi nào KHÁC KeyboardInterrupt (ví dụ RuntimeError từ 1 bước
trong pipeline - Microphone/STT/Normalize/chatbot/TTS/Speaker) đều
KHÔNG bị bắt ở đây và sẽ làm chương trình dừng với traceback đầy đủ -
đúng tinh thần "không nuốt exception" đã xuyên suốt từ voice_chat.py:
lỗi thật sự (không phải người dùng chủ động dừng) nên dừng chương trình
và hiện rõ nguyên nhân, không nên bị nuốt để vòng lặp "giả vờ" tiếp tục
chạy.
"""
from __future__ import annotations

import logging

from voice.voice_chat import VoiceChat

# Cấu hình logging cơ bản để các log INFO đã có sẵn trong voice_chat.py/
# microphone.py/stt.py/tts.py/speaker.py ("Recording...", "Transcribing...",
# "Done."...) thực sự hiện ra console khi chạy script này - logging module
# mặc định không có handler nào nên các log INFO sẽ bị ẩn hoàn toàn nếu
# thiếu dòng basicConfig này. Đây KHÔNG phải CLI parser/menu, chỉ là cấu
# hình logging tối thiểu cho 1 entry point chạy độc lập.
logging.basicConfig(level=logging.INFO)

RECORDING_DURATION_SECONDS = 5

if __name__ == "__main__":
    voice_chat = VoiceChat()
    while True:
        try:
            voice_chat.listen_once(duration=RECORDING_DURATION_SECONDS)
        except KeyboardInterrupt:
            break
