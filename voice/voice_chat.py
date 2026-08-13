"""
voice/voice_chat.py
Orchestrator: nối các module đã có của Voice Chat thành đúng 1 pipeline:

    Microphone.record()
        -> SpeechToText.transcribe()
        -> TextNormalizer.normalize()
        -> chatbot.ask()
        -> TextToSpeech.synthesize()
        -> Speaker.play()

File này KHÔNG implement bất kỳ logic nghiệp vụ nào của từng bước - toàn
bộ logic ghi âm/STT/normalize/sinh câu trả lời/TTS/phát loa đã nằm sẵn
trong các module tương ứng (microphone.py, stt.py, text_normalizer.py,
chatbot.py, tts.py, speaker.py). `VoiceChat` chỉ GỌI ĐÚNG THỨ TỰ các
module đó và chuyển output của bước này thành input của bước sau - đúng
nghĩa đen của "orchestrator": điều phối, không thực thi.

`VoiceChat` KHÔNG biết bên trong `chatbot.ask()` có RAG, có Memory, có
Tool Calling hay không - nó chỉ biết `chatbot.ask(text: str) -> str`.
Đây chính là lý do voice_chat.py CHỈ import `chatbot.ask`, KHÔNG import
`rag`, `memory`, `llm`, `tools`, `tool_executor` - những chi tiết đó đã
được `chatbot.py` đóng gói (encapsulate) sẵn, orchestrator ở tầng voice/
không có lý do gì phải biết tới chúng.

`listen_once()` xử lý ĐÚNG MỘT lượt hội thoại (ghi âm -> ... -> phát trả
lời), không có vòng lặp, không có CLI, không có `input()`/`print()`.
Việc lặp lại nhiều lượt (ví dụ vòng lặp "nói chuyện liên tục") là quyết
định của chương trình gọi `VoiceChat` (ví dụ 1 script `main.py` ở ngoài
package `voice/`), không thuộc trách nhiệm của orchestrator này.
"""
from __future__ import annotations

import logging
import time

from chatbot import ask
from voice.microphone import Microphone
from voice.speaker import Speaker
from voice.stt import SpeechToText
from voice.text_normalizer import TextNormalizer
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceChat:
    """Điều phối 1 lượt hội thoại thoại đầy đủ qua 6 bước:
    Microphone -> SpeechToText -> TextNormalizer -> chatbot.ask ->
    TextToSpeech -> Speaker.

    Mỗi bước là 1 dependency được khởi tạo (hoặc inject) trong
    `__init__`, KHÔNG có global state/singleton nào - mỗi instance
    `VoiceChat` tự chứa toàn bộ dependency của riêng nó.

    Các dependency (`microphone`, `stt`, `normalizer`, `tts`, `speaker`)
    có thể được TRUYỀN VÀO qua `__init__` thay vì để `VoiceChat` tự khởi
    tạo. Đây là cách duy nhất để unit test `listen_once()` mà không cần
    microphone/loa thật hay gọi mạng thật tới LLM/edge-tts: test chỉ cần
    truyền vào 5 đối tượng giả (mock/stub) khớp đúng API công khai của
    `Microphone`/`SpeechToText`/`TextNormalizer`/`TextToSpeech`/`Speaker`,
    và không cần mock `chatbot.ask` theo cách phức tạp hơn so với
    `unittest.mock.patch("voice.voice_chat.ask", ...)`.
    """

    def __init__(
        self,
        microphone: Microphone | None = None,
        stt: SpeechToText | None = None,
        normalizer: TextNormalizer | None = None,
        tts: TextToSpeech | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        """Khởi tạo (hoặc nhận sẵn) toàn bộ dependency của pipeline.

        Args:
            microphone: instance `Microphone` dùng để ghi âm. Nếu
                `None`, tự khởi tạo `Microphone()`.
            stt: instance `SpeechToText` dùng để chuyển audio -> text.
                Nếu `None`, tự khởi tạo `SpeechToText()`.
            normalizer: instance `TextNormalizer` dùng để chuẩn hóa text
                RAW từ STT. Nếu `None`, tự khởi tạo `TextNormalizer()`.
            tts: instance `TextToSpeech` dùng để chuyển câu trả lời của
                chatbot -> audio bytes. Nếu `None`, tự khởi tạo
                `TextToSpeech()`.
            speaker: instance `Speaker` dùng để phát audio ra loa. Nếu
                `None`, tự khởi tạo `Speaker()`.

        Raises:
            RuntimeError: lan truyền từ `Microphone()`, `SpeechToText()`,
                `TextToSpeech()`, hoặc `Speaker()` nếu VOICE_ENABLED=False
                hoặc thiết bị/model không khởi tạo được - `VoiceChat`
                không tự bắt các lỗi này, để nguyên như các module con đã
                raise (fail-fast ngay lúc khởi tạo, không đợi tới
                `listen_once()`).
        """
        self._microphone = microphone if microphone is not None else Microphone()
        self._stt = stt if stt is not None else SpeechToText()
        self._normalizer = normalizer if normalizer is not None else TextNormalizer()
        self._tts = tts if tts is not None else TextToSpeech()
        self._speaker = speaker if speaker is not None else Speaker()

    def listen_once(self, duration: float) -> str:
        """Thực hiện đúng MỘT lượt hội thoại thoại đầy đủ: ghi âm, nhận
        dạng giọng nói, chuẩn hóa text, hỏi chatbot, tổng hợp giọng nói
        cho câu trả lời, rồi phát ra loa.

        Đây là API dạng "1 lượt" (không phải vòng lặp) - gọi hàm này
        nhiều lần để có nhiều lượt hội thoại là quyết định của caller,
        không phải của `VoiceChat`.

        Args:
            duration: thời lượng ghi âm, tính bằng giây. Được chuyển
                thẳng cho `Microphone.record(duration)`.

        Returns:
            Câu trả lời dạng text (chính là giá trị `chatbot.ask()` trả
            về) - trả về DẠNG TEXT để caller vẫn có thể hiển thị/log câu
            trả lời, dù âm thanh đã được phát ra loa trong cùng lượt gọi
            này.

        Raises:
            RuntimeError: nếu bất kỳ bước nào trong pipeline lỗi (ghi
                âm, STT, normalize, chatbot, TTS, phát loa). Lỗi gốc
                (exception ban đầu) được log đầy đủ bằng `logger.exception`
                trước khi raise lại dưới dạng `RuntimeError` có ngữ cảnh
                cho biết đang lỗi ở bước nào - không có bước nào bị nuốt
                (swallow) exception.
        """
        # Đo thời gian TOÀN BỘ pipeline (mốc bắt đầu, không tính vào bước
        # nào cụ thể) - dùng time.perf_counter() thay vì time.time() vì
        # perf_counter là đồng hồ đơn điệu (monotonic), không bị ảnh hưởng
        # bởi việc hệ thống chỉnh giờ (NTP sync...) giữa lúc đo, phù hợp để
        # đo khoảng thời gian (duration) thay vì mốc thời gian tuyệt đối.
        pipeline_start = time.perf_counter()

        logger.info("Recording...")
        step_start = time.perf_counter()
        try:
            audio = self._microphone.record(duration)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước Microphone.record()")
            raise RuntimeError("VoiceChat: lỗi ở bước ghi âm (Microphone)") from exc
        logger.info("Recording completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Transcribing...")
        step_start = time.perf_counter()
        try:
            raw_text = self._stt.transcribe(audio)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước SpeechToText.transcribe()")
            raise RuntimeError("VoiceChat: lỗi ở bước nhận dạng giọng nói (STT)") from exc
        logger.info("STT completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Normalizing...")
        step_start = time.perf_counter()
        try:
            normalized_text = self._normalizer.normalize(raw_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TextNormalizer.normalize()")
            raise RuntimeError("VoiceChat: lỗi ở bước chuẩn hóa text") from exc
        logger.info("Normalize completed in %.2fs", time.perf_counter() - step_start)

        # không phát hiện giọng nói (STT trả về chuỗi rỗng hoặc chỉ toàn khoảng trắng)
        if not normalized_text.strip():
            logger.info("Không phát hiện giọng nói, bỏ qua lượt hội thoại.")
            logger.info(
                "Pipeline completed in %.2fs",
                time.perf_counter() - pipeline_start,
            )
            return ""
        
        logger.info("Generating answer...")
        step_start = time.perf_counter()
        try:
            answer = ask(normalized_text)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước chatbot.ask()")
            raise RuntimeError("VoiceChat: lỗi ở bước sinh câu trả lời (chatbot)") from exc
        logger.info("LLM completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Synthesizing...")
        step_start = time.perf_counter()
        try:
            audio_bytes = self._tts.synthesize(answer)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước TextToSpeech.synthesize()")
            raise RuntimeError("VoiceChat: lỗi ở bước tổng hợp giọng nói (TTS)") from exc
        logger.info("TTS completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Playing...")
        step_start = time.perf_counter()
        try:
            self._speaker.play(audio_bytes)
        except Exception as exc:
            logger.exception("VoiceChat lỗi ở bước Speaker.play()")
            raise RuntimeError("VoiceChat: lỗi ở bước phát audio (Speaker)") from exc
        logger.info("Playback completed in %.2fs", time.perf_counter() - step_start)

        logger.info("Pipeline completed in %.2fs", time.perf_counter() - pipeline_start)
        return answer
