"""
voice/microphone.py
Microphone: thiết bị thu âm -> numpy.ndarray. Đây là bước ĐẦU TIÊN trong
pipeline Voice Chat, đứng TRƯỚC stt.py:

    [microphone.py] -> Audio (np.ndarray) -> stt.py -> text_normalizer.py
        -> chatbot.ask() -> tts.py -> speaker.py

File này CHỈ có đúng 1 trách nhiệm: thu âm trong N giây, trả về waveform
dạng numpy array. Không làm gì khác:
- Không lưu file (không ghi .wav ra đĩa).
- Không gọi faster-whisper / STT - đó là việc của stt.py, đứng ở lớp
  ngoài microphone.py, microphone.py không import stt.
- Không normalize text - microphone.py còn chưa hề có "text", chỉ có
  audio thô.
- Không biết gì về chatbot/RAG/Memory/Tool Calling/LLM/TTS/Speaker -
  không import bất kỳ module nào trong số đó.
- Không hardcode nghiệp vụ (không có logic menu/promotions/business gì ở
  đây, đây thuần túy là I/O phần cứng).

THƯ VIỆN: `sounddevice` (bọc PortAudio). Chọn vì:
- API đơn giản nhất cho use-case "thu âm N giây, trả numpy array":
  `sd.rec(...)` trả thẳng `np.ndarray`, không cần tự quản lý callback,
  buffer, hay stream thủ công như khi dùng `pyaudio`.
- Không cần ghi file trung gian (khác nhiều ví dụ dùng `wave` module) -
  khớp đúng yêu cầu "không lưu file" của phase này.
- Cùng hệ sinh thái numpy mà stt.py (Phase STT) đã dùng, nên
  `Microphone.record()` trả ra đúng kiểu `np.ndarray` mà
  `SpeechToText.transcribe()` cần, không cần lớp chuyển đổi ở giữa.

VỀ THIẾT BỊ GHI ÂM: `VoiceConfig.device` là compute device cho model
Whisper ("cpu"/"cuda" - xem stt.py), KHÔNG PHẢI audio input device vật
lý (microphone nào trên máy). Hai khái niệm "device" này khác nhau hoàn
toàn dù trùng tên field. Vì vậy microphone.py KHÔNG dùng
`config.device` - `sounddevice` sẽ dùng microphone MẶC ĐỊNH của hệ điều
hành. Nếu sau này cần chọn 1 microphone cụ thể (nhiều micro cùng lúc),
cần thêm field mới vào VoiceConfig - xem đề xuất ở cuối câu trả lời,
KHÔNG tái sử dụng `config.device` cho việc này.
"""
from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from voice.config import get_voice_config

logger = logging.getLogger(__name__)

_CHANNELS = 1
_DTYPE = "float32"


class Microphone:
    """Bọc `sounddevice`, chỉ expose đúng 1 hành vi: thu âm N giây ->
    numpy array (mono, float32).

    Không giữ state giữa các lần gọi `record()` (không buffer, không
    cache) - mỗi lần gọi là một lượt thu âm độc lập, tự chứa. Điều này
    giúp `Microphone` dễ test độc lập: test chỉ cần mock `sounddevice`,
    không phụ thuộc `chatbot`/`stt`/bất kỳ module nào khác.
    """

    def __init__(self) -> None:
        """Đọc cấu hình từ voice.config.get_voice_config() và kiểm tra
        có microphone khả dụng hay không.

        Raises:
            RuntimeError: nếu VOICE_ENABLED=False (không nên khởi tạo
                Microphone khi Voice đang tắt), hoặc nếu không tìm thấy
                microphone khả dụng trên hệ thống (không có thiết bị
                input, hoặc thiết bị input mặc định không hỗ trợ
                sample_rate/số kênh yêu cầu).
        """
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "Microphone được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo Microphone khi Voice đang tắt."
            )

        self._sample_rate = config.sample_rate

        try:
            sd.check_input_settings(
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
            )
        except Exception as exc:
            logger.exception(
                "Không tìm thấy microphone khả dụng (sample_rate=%d Hz, channels=%d)",
                self._sample_rate,
                _CHANNELS,
            )
            raise RuntimeError(
                "Không tìm thấy microphone khả dụng trên hệ thống"
            ) from exc

        logger.info("Đã khởi tạo Microphone (sample_rate=%d Hz)", self._sample_rate)

    def record(self, duration: float) -> np.ndarray:
        """Thu âm từ microphone mặc định của hệ thống trong `duration`
        giây.

        Args:
            duration: thời lượng cần thu âm, tính bằng giây, phải > 0.

        Returns:
            Waveform mono, dtype float32, biên độ trong khoảng
            [-1.0, 1.0], mảng 1 chiều (đã squeeze bỏ chiều channel), thu
            ở sample rate = VOICE_SAMPLE_RATE (config.sample_rate) -
            đúng định dạng mà `SpeechToText.transcribe()` (stt.py) yêu
            cầu.

        Raises:
            ValueError: nếu `duration` không hợp lệ (không phải số,
                hoặc <= 0).
            RuntimeError: nếu quá trình thu âm thất bại (microphone bị
                rút ra giữa chừng, lỗi driver PortAudio, v.v.).
        """
        self._validate_duration(duration)

        num_frames = int(duration * self._sample_rate)
        logger.info(
            "Bắt đầu ghi âm %.2fs (%d mẫu, %d Hz)",
            duration,
            num_frames,
            self._sample_rate,
        )

        try:
            audio = sd.rec(
                num_frames,
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype=_DTYPE,
            )
            sd.wait()
        except Exception as exc:
            logger.exception("Lỗi khi ghi âm từ microphone (duration=%.2fs)", duration)
            raise RuntimeError("Không thể ghi âm từ microphone") from exc

        # sd.rec() trả về mảng shape (num_frames, channels); channels=1
        # nên squeeze về mảng 1 chiều (num_frames,) - đúng shape mà
        # stt.py (_validate_audio: audio.ndim != 1) yêu cầu.
        waveform = audio.reshape(-1).astype(np.float32)

        logger.info("Ghi âm xong: %d mẫu (%.2fs)", waveform.shape[0], duration)
        return waveform

    @staticmethod
    def _validate_duration(duration: float) -> None:
        """Kiểm tra `duration` hợp lệ trước khi thu âm, để lỗi (nếu có)
        báo ngay tại đây với message rõ ràng, thay vì để sounddevice ném
        ra lỗi khó hiểu (hoặc tệ hơn: thu âm 0 giây/số âm một cách im
        lặng)."""
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError(
                f"duration phải là số (int/float), nhận được: {type(duration)!r}"
            )
        if duration <= 0:
            raise ValueError(f"duration phải > 0, hiện tại: {duration}")
