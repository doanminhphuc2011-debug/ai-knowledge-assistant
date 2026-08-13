"""
voice/speaker.py
Speaker: audio bytes -> phát ra loa. Đây là bước CUỐI CÙNG trong pipeline
Voice Chat, đứng SAU tts.py:

    ... -> chatbot.ask() -> tts.py -> Audio bytes -> [speaker.py] -> Loa

File này CHỈ có đúng 1 trách nhiệm: nhận audio bytes, phát ra loa. Không
làm gì khác:
- Không decode/diễn giải text - speaker.py không hề biết nội dung câu
  nói là gì, chỉ biết "đây là audio, phát nó".
- Không biết gì về chatbot/RAG/Memory/Tool Calling/LLM/STT/Microphone -
  không import bất kỳ module nào trong số đó.
- Không ghi file ra đĩa (không lưu lại audio đã phát).
- Không hardcode nghiệp vụ - thuần túy I/O phần cứng.

VỀ ĐỊNH DẠNG AUDIO ĐẦU VÀO: theo thiết kế của tts.py (Phase TTS),
`TextToSpeech.synthesize()` trả về đúng bytes mà provider (edge-tts) tạo
ra, KHÔNG bị ép về 1 định dạng cố định. speaker.py vì vậy phải TỰ NHẬN
DIỆN định dạng từ chính nội dung bytes.

THƯ VIỆN: đã bỏ hẳn `pydub` + `simpleaudio` (không tương thích Python
3.13+, vì `pydub` import module chuẩn `audioop` đã bị loại khỏi Python
3.13). Thay bằng đúng 3 thư viện: `sounddevice`, `soundfile`, `numpy`.

Luồng xử lý trong `play()`:
1. `soundfile.read()` decode audio bytes (qua `io.BytesIO`) ở các định
   dạng mà `libsndfile` hỗ trợ (wav, flac, ogg/vorbis, ...), trả thẳng
   về mảng numpy `float32` đã chuẩn hóa biên độ về khoảng [-1.0, 1.0] -
   không cần tự tính `sample_width`/`max_amplitude` như trước, vì
   `soundfile` làm việc này khi đọc với `dtype="float32"`.
2. `sd.play(data, sample_rate)` phát mảng numpy đó, `sd.wait()` block
   cho tới khi phát xong (cần thiết vì `play()` phải block tới khi phát
   xong, không phải "bắn" audio rồi trả về ngay trong khi loa còn đang
   phát).

LƯU Ý: `soundfile`/`libsndfile` KHÔNG decode được mp3 trên một số nền
tảng/bản dựng libsndfile cũ - nhưng khác với `pydub`, thư viện này
KHÔNG cần cài thêm `ffmpeg` (không phải gói pip) trên hệ thống, nên
`__init__` không cần kiểm tra `ffmpeg` nữa.

Không giữ state giữa các lần gọi `play()` (không buffer, không cache) -
mỗi lần gọi là một lượt phát âm thanh độc lập, tự chứa.
"""
from __future__ import annotations

import io
import logging

import soundfile as sf
import sounddevice as sd

logger = logging.getLogger(__name__)


class Speaker:
    """Bọc `soundfile` (decode) + `sounddevice` (playback), chỉ expose
    đúng 1 hành vi: audio bytes -> phát ra loa, block tới khi phát xong.

    Không giữ state giữa các lần gọi `play()` (không buffer, không
    cache) - mỗi lần gọi là một lượt phát âm thanh độc lập, tự chứa.
    Điều này giúp `Speaker` dễ test độc lập: test chỉ cần mock
    `soundfile`/`sounddevice`, không phụ thuộc `chatbot`/`tts`/bất kỳ
    module nào khác.
    """

    def __init__(self) -> None:
        """Đọc cấu hình từ voice.config.get_voice_config() và kiểm tra
        có thiết bị phát âm thanh (loa) khả dụng hay không - kiểm tra
        NGAY LÚC KHỞI TẠO (fail-fast), không đợi tới lúc `play()` mới
        phát hiện.

        Việc kiểm tra thiết bị loa dùng
        `sounddevice.check_output_settings()` (chỉ để hỏi PortAudio "có
        output device mặc định không"). Cùng thư viện `sounddevice` này
        sau đó cũng được dùng để phát audio thực sự trong `play()`.

        Raises:
            RuntimeError: nếu VOICE_ENABLED=False (không nên khởi tạo
                Speaker khi Voice đang tắt), hoặc nếu không tìm thấy
                thiết bị phát âm thanh khả dụng trên hệ thống.
        """
        # Import cục bộ trong __init__ để tránh vòng import không cần
        # thiết ở mức module - get_voice_config() chỉ cần khi khởi tạo,
        # play() không cần đọc lại config mỗi lần phát.
        from voice.config import get_voice_config

        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "Speaker được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo Speaker khi Voice đang tắt."
            )

        try:
            sd.check_output_settings()
        except Exception as exc:
            logger.exception("Không tìm thấy thiết bị phát âm thanh (loa) khả dụng")
            raise RuntimeError(
                "Không tìm thấy thiết bị phát âm thanh (loa) khả dụng trên hệ thống"
            ) from exc

        logger.info("Đã khởi tạo Speaker")

    def play(self, audio: bytes) -> None:
        """Decode `audio` và phát ra loa, block cho tới khi phát xong.

        Args:
            audio: audio bytes cần phát, ví dụ output của
                `TextToSpeech.synthesize()`. speaker.py tự nhận diện
                định dạng từ chính nội dung bytes, không giả định trước
                là wav/ogg/...

        Returns:
            None.

        Raises:
            ValueError: nếu `audio` không hợp lệ (không phải bytes, hoặc
                rỗng - không có gì để phát).
            RuntimeError: nếu không decode được `audio` (dữ liệu hỏng,
                định dạng không được `soundfile` hỗ trợ), hoặc nếu phát
                thất bại (lỗi thiết bị loa, driver PortAudio).
        """
        self._validate_audio(audio)

        try:
            data, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
        except Exception as exc:
            logger.exception("Không thể decode audio bytes (độ dài=%d byte)", len(audio))
            raise RuntimeError(
                "Không thể decode audio bytes - dữ liệu hỏng hoặc định "
                "dạng không được hỗ trợ"
            ) from exc

        duration_seconds = len(data) / float(sample_rate) if sample_rate else 0.0
        channels = data.shape[1] if data.ndim > 1 else 1

        logger.info(
            "Bắt đầu phát audio: %.2fs, %d Hz, %d kênh",
            duration_seconds,
            sample_rate,
            channels,
        )

        try:
            sd.play(data, sample_rate)
            sd.wait()
        except Exception as exc:
            logger.exception("Lỗi khi phát audio qua loa")
            raise RuntimeError("Không thể phát audio qua loa") from exc

        logger.info("Phát audio xong: %.2fs", duration_seconds)

    @staticmethod
    def _validate_audio(audio: bytes) -> None:
        """Kiểm tra `audio` hợp lệ trước khi decode/phát, để lỗi (nếu
        có) báo ngay tại đây với message rõ ràng, thay vì để
        soundfile/sounddevice ném ra lỗi khó hiểu ở tầng sâu hơn."""
        if not isinstance(audio, bytes):
            raise ValueError(f"audio phải là bytes, nhận được: {type(audio)!r}")
        if len(audio) == 0:
            raise ValueError("audio rỗng, không có gì để phát")
