"""
voice/stt.py
Speech-to-Text: Audio -> Text. Đây là bước ĐẦU TIÊN trong pipeline Voice
Chat (đứng trước text_normalizer.py và chatbot.ask()):

    Audio -> [stt.py] -> Text (raw) -> text_normalizer.py -> chatbot.ask()

File này CHỈ có đúng 1 trách nhiệm: nhận audio, trả về text. Không làm gì
khác:
- Không normalize text (đó là việc của text_normalizer.py ở phase sau).
- Không biết gì về chatbot/RAG/Retriever/Redis/Memory/Tool Calling/LLM -
  những thứ đó chỉ tồn tại phía sau `from chatbot import ask`, và stt.py
  không import chatbot.
- Không thu âm (microphone), không phát âm (speaker), không đo latency.

Dùng faster-whisper (CTranslate2 re-implementation của Whisper) thay vì
thư viện `openai-whisper` gốc, vì nhanh hơn đáng kể và tốn ít bộ nhớ hơn
với cùng độ chính xác - phù hợp hơn cho 1 voice chat cần phản hồi nhanh.

LƯU Ý VỀ SAMPLE RATE (ràng buộc của engine, không phải business logic):
faster-whisper yêu cầu audio đầu vào dạng mảng (np.ndarray) phải là mono,
16000Hz - đây là ràng buộc của bản thân model Whisper, áp dụng cho MỌI
checkpoint. stt.py không tự kiểm tra/ép giá trị này trong code; nó chỉ
dùng đúng VOICE_SAMPLE_RATE lấy từ VoiceConfig. Việc đảm bảo
VOICE_SAMPLE_RATE=16000 trong .env là trách nhiệm của cấu hình/vận hành,
không phải của STT.
"""
from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from voice.config import get_voice_config

logger = logging.getLogger(__name__)


class SpeechToText:
    """Bọc `faster_whisper.WhisperModel`, chỉ expose đúng 1 hành vi:
    audio (mảng waveform đã decode sẵn) -> text.

    Model được tải MỘT LẦN trong __init__ (việc tải model tốn vài giây và
    chiếm RAM/VRAM đáng kể) - caller (voice_chat.py, ở phase sau) nên giữ
    1 instance duy nhất dùng xuyên suốt vòng đời ứng dụng, thay vì tạo mới
    SpeechToText() cho mỗi lượt hội thoại.
    """

    def __init__(self) -> None:
        """Đọc cấu hình từ voice.config.get_voice_config() và tải model.

        Raises:
            RuntimeError: nếu VOICE_ENABLED=False (không nên khởi tạo STT
                khi Voice đang tắt), hoặc nếu faster-whisper không tải
                được model.
        """
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "SpeechToText được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo STT khi Voice đang tắt."
            )

        self._language = config.language
        self._sample_rate = config.sample_rate

        logger.info(
            "Đang tải faster-whisper model='%s' trên device='%s'...",
            config.model,
            config.device,
        )
        try:
            # compute_type KHÔNG được truyền vào đây vì VoiceConfig hiện
            # chưa có field này (xem giải thích ở cuối câu trả lời) - để
            # faster-whisper tự chọn compute_type mặc định phù hợp với
            # device, thay vì stt.py tự hardcode 1 giá trị cụ thể.
            self._model = WhisperModel(config.model, device=config.device)
        except Exception as exc:
            logger.exception(
                "Không thể tải faster-whisper model='%s' trên device='%s'",
                config.model,
                config.device,
            )
            raise RuntimeError(
                f"Không thể khởi tạo faster-whisper model='{config.model}' "
                f"trên device='{config.device}'"
            ) from exc

        logger.info("Đã tải xong faster-whisper model='%s'", config.model)

    def transcribe(self, audio: np.ndarray) -> str:
        """Chuyển audio (đã decode sẵn thành waveform) thành text.

        Đây là API dạng mảng (np.ndarray) thay vì đường dẫn file, vì Voice
        Chat là một vòng lặp hội thoại theo thời gian thực (record -> STT
        -> chatbot.ask() -> TTS -> phát) - ghi audio ra file rồi đọc lại
        cho mỗi lượt nói sẽ tốn thêm I/O đĩa và độ trễ không cần thiết.
        Việc thu âm/decode audio thành mảng thuộc trách nhiệm của
        voice_chat.py (phase sau); stt.py chỉ nhận mảng đã sẵn sàng.

        Args:
            audio: waveform mono, dtype float32, biên độ trong khoảng
                [-1.0, 1.0], được thu ở sample rate = VOICE_SAMPLE_RATE
                (config.sample_rate). Xem module docstring về ràng buộc
                sample rate của engine faster-whisper.

        Returns:
            Văn bản nhận dạng được, RAW - CHƯA qua normalize (đó là việc
            của text_normalizer.py). Trả về chuỗi rỗng "" nếu không nhận
            ra được lời nói nào trong audio.

        Raises:
            ValueError: nếu `audio` không hợp lệ (không phải np.ndarray,
                không phải mảng 1 chiều, rỗng, hoặc sai dtype).
            RuntimeError: nếu faster-whisper lỗi trong lúc transcribe.
        """
        self._validate_audio(audio)

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
            )
            # segments là generator - việc iterate chính là lúc model
            # thực sự chạy suy luận (lazy evaluation của faster-whisper).
            text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            logger.exception(
                "Lỗi khi transcribe audio (%d mẫu, %.2fs)",
                audio.shape[0],
                audio.shape[0] / self._sample_rate,
            )
            raise RuntimeError("Không thể transcribe audio") from exc

        logger.info(
            "Transcribe xong: %.2fs audio, ngôn ngữ phát hiện='%s' "
            "(prob=%.2f), độ dài text=%d ký tự",
            audio.shape[0] / self._sample_rate,
            info.language,
            info.language_probability,
            len(text),
        )
        return text

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        """Kiểm tra audio đầu vào đúng định dạng trước khi đưa vào model,
        để lỗi (nếu có) báo ngay tại đây với message rõ ràng, thay vì để
        faster-whisper ném ra 1 lỗi khó hiểu ở tầng sâu hơn."""
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"audio phải là np.ndarray, nhận được: {type(audio)!r}")
        if audio.ndim != 1:
            raise ValueError(
                f"audio phải là mảng 1 chiều (mono), nhận được ndim={audio.ndim}"
            )
        if audio.size == 0:
            raise ValueError("audio rỗng, không có gì để transcribe")
        if audio.dtype != np.float32:
            raise ValueError(
                f"audio phải có dtype=float32, nhận được: {audio.dtype}"
            )
