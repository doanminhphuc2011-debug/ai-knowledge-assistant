"""
voice/tts.py
Text-to-Speech: Text -> Audio bytes. Đây là bước GẦN CUỐI trong pipeline
Voice Chat (đứng sau chatbot.ask(), trước Speaker - Speaker KHÔNG thuộc
phạm vi file này):

    chatbot.ask() -> Text (câu trả lời) -> [tts.py] -> Audio bytes -> Speaker

File này CHỈ có đúng 1 trách nhiệm: nhận text, trả về audio bytes. Không
làm gì khác:
- Không phát loa (đó là việc của Speaker/voice_chat.py ở phase sau).
- Không ghi file ra đĩa - caller tự quyết định làm gì với bytes trả về
  (phát trực tiếp, lưu file, stream qua mạng...).
- Không ép/validate định dạng audio provider trả về (mp3/wav/ogg/...).
  tts.py chỉ quan tâm text -> bytes; Speaker ở phase sau mới là nơi biết
  cách phát đúng định dạng mà provider thực sự trả về.
- Không normalize/đọc số/bỏ dấu/sửa câu - text đưa vào synthesize() thế
  nào thì đọc y nguyên thế đó (đó là việc của text_normalizer.py, và chỉ
  áp dụng cho input của STT, không áp dụng cho output của chatbot).
- Không biết gì về chatbot/RAG/Retriever/Redis/Memory/Tool Calling/LLM -
  tts.py không import chatbot, không import rag, không import memory.
- Không biết microphone/speaker.

PROVIDER: edge-tts (Microsoft Edge neural TTS, dùng qua thư viện Python
`edge-tts`, không cần API key). Chọn edge-tts vì:
- Miễn phí, không cần đăng ký API key (khác Azure Cognitive Services trả
  phí) - phù hợp với 1 project chatbot đang ở giai đoạn phát triển.
- Hỗ trợ tiếng Việt chất lượng tốt (giọng neural).
- Thư viện async-first, gọn nhẹ, không kéo theo dependency nặng.

Thiết kế để CÓ THỂ thêm provider khác sau này (ví dụ: Google TTS, Azure
TTS, ElevenLabs...) mà không phải viết lại từ đầu: toàn bộ lời gọi tới
edge-tts được cô lập trong đúng 1 method private (`_call_provider`).
Muốn đổi provider, chỉ cần thay nội dung method đó. KHÔNG tạo abstract
base class, KHÔNG tạo provider registry/factory, KHÔNG tạo plugin system
ở phase này - hiện tại chỉ có đúng 1 provider, thêm các lớp trừu tượng đó
bây giờ là over-engineering (YAGNI). Khi nào thực sự có provider thứ 2,
hãy refactor lúc đó.

VỀ TÊN GIỌNG ĐỌC (voice) CỦA TTS:
`VoiceConfig.language` (ví dụ "vi") là mã ngôn ngữ NGẮN dành riêng cho
STT (faster-whisper dùng mã kiểu ISO ngắn gọn: "vi", "en"...). edge-tts
lại cần TÊN GIỌNG ĐẦY ĐỦ, khác định dạng hoàn toàn (ví dụ
"vi-VN-HoaiMyNeural"). Hai giá trị này KHÔNG tương thích và KHÔNG được
dùng lẫn cho nhau - dùng "vi" làm tên giọng edge-tts sẽ lỗi ngay khi gọi
provider.

`VoiceConfig` HIỆN CHƯA có field nào cho việc này, và tts.py KHÔNG tự ý
sửa config.py để thêm field. Vì vậy tts.py đọc trực tiếp biến môi trường
`VOICE_TTS_VOICE` (qua os.getenv, KHÔNG qua get_voice_config()) như một
giải pháp TẠM THỜI, tách biệt hoàn toàn khỏi `VoiceConfig.language`. Đây
là ngoại lệ có chủ đích, không phải tùy tiện đọc os.getenv - xem đề xuất
bổ sung field ở cuối câu trả lời để đưa việc đọc biến này vào đúng chỗ
(VoiceConfig) trong 1 lần sửa config.py sau này.
"""
from __future__ import annotations

import asyncio
import logging
import os

import edge_tts
import edge_tts.exceptions

from voice.config import get_voice_config

logger = logging.getLogger(__name__)

_TTS_VOICE_ENV_VAR = "VOICE_TTS_VOICE"

# RETRY CHO edge-tts: `edge_tts.exceptions.NoAudioReceived` thường là lỗi
# TRANSIENT (WebSocket của endpoint Microsoft bị đóng giữa chừng, hoặc
# rate-limit tạm thời) chứ không phải lỗi cấu hình - retry ngay lập tức có
# xác suất thành công cao. 3 lần thử là đủ để vượt qua glitch mạng thoáng
# qua mà không làm người dùng chờ quá lâu nếu provider thực sự đang down.
_TTS_MAX_RETRIES = 3
# Backoff mũ (exponential): 1s -> 2s -> 4s. Bắt đầu từ 1s (không phải
# mili-giây) vì nguyên nhân phổ biến nhất là rate-limit phía server, cần vài
# giây để "nguội" trước khi thử lại, thử lại ngay lập tức (0ms) dễ bị chính
# lỗi đó lặp lại liên tiếp.
_TTS_RETRY_BASE_DELAY_SECONDS = 1.0


class TextToSpeech:
    """Bọc `edge_tts.Communicate`, chỉ expose đúng 1 hành vi:
    text -> audio bytes.

    Không giữ audio đã sinh ra trong bộ nhớ giữa các lần gọi, không có
    cache - mỗi lần gọi `synthesize()` là một lượt tổng hợp giọng nói độc
    lập, tự chứa (self-contained), không phụ thuộc lượt gọi trước.
    """

    def __init__(self) -> None:
        """Đọc cấu hình Voice (qua get_voice_config()) và tên giọng đọc
        TTS (qua biến môi trường VOICE_TTS_VOICE - xem module docstring
        về lý do KHÔNG lấy qua VoiceConfig.language).

        Raises:
            RuntimeError: nếu VOICE_ENABLED=False (không nên khởi tạo TTS
                khi Voice đang tắt), hoặc nếu thiếu biến môi trường
                VOICE_TTS_VOICE (bắt buộc phải có để gọi edge-tts, xem
                đề xuất bổ sung field ở cuối câu trả lời).
        """
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "TextToSpeech được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo TTS khi Voice đang tắt."
            )

        voice = os.getenv(_TTS_VOICE_ENV_VAR)
        if voice is None or voice.strip() == "":
            raise RuntimeError(
                f"Thiếu biến môi trường {_TTS_VOICE_ENV_VAR} trong .env. "
                f"tts.py cần tên giọng đọc edge-tts đầy đủ (ví dụ "
                f"'vi-VN-HoaiMyNeural'), KHÔNG dùng VOICE_LANGUAGE vì "
                f"biến đó dành riêng cho STT. Xem đề xuất bổ sung field "
                f"vào VoiceConfig ở cuối câu trả lời."
            )
        self._voice = voice.strip()

        logger.info("Đã khởi tạo TextToSpeech (provider=edge-tts, voice='%s')", self._voice)

    def synthesize(self, text: str) -> bytes:
        """Chuyển text thành audio bytes.

        Args:
            text: câu cần đọc thành giọng nói. Được đưa thẳng vào provider,
                KHÔNG qua bất kỳ bước normalize/đọc số/sửa câu nào ở đây.

        Returns:
            Audio bytes do provider (edge-tts) trả về, giữ nguyên định
            dạng gốc của provider - tts.py không ép/chuyển đổi định dạng.
            Caller (Speaker ở phase sau) chịu trách nhiệm biết và phát
            đúng định dạng đó.

        Raises:
            ValueError: nếu `text` không hợp lệ (không phải str, hoặc chỉ
                toàn khoảng trắng/rỗng - không có gì để đọc).
            RuntimeError: nếu provider edge-tts lỗi trong lúc tổng hợp
                giọng nói (mạng lỗi, dịch vụ từ chối, tên giọng không tồn
                tại, v.v.), hoặc nếu provider không trả về byte audio nào.
        """
        self._validate_text(text)

        try:
            audio_bytes = asyncio.run(self._call_provider(text))
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception(
                "Lỗi khi tổng hợp giọng nói (voice='%s', độ dài text=%d ký tự)",
                self._voice,
                len(text),
            )
            raise RuntimeError("Không thể tổng hợp giọng nói (TTS)") from exc

        if not audio_bytes:
            logger.error(
                "edge-tts không trả về audio nào cho voice='%s', độ dài text=%d ký tự",
                self._voice,
                len(text),
            )
            raise RuntimeError("Provider edge-tts không trả về audio nào")

        logger.info(
            "Tổng hợp giọng nói xong: %d ký tự text -> %d byte audio",
            len(text),
            len(audio_bytes),
        )
        return audio_bytes

    async def _call_provider(self, text: str) -> bytes:
        """Gọi edge-tts và gom các chunk audio trả về thành 1 bytes duy
        nhất. Toàn bộ chi tiết riêng của provider edge-tts (async
        streaming API, cấu trúc chunk dict, định dạng audio provider tự
        chọn...) bị cô lập trong đúng method này - đổi provider sau này
        chỉ cần viết lại method này.

        Tự động RETRY tối đa `_TTS_MAX_RETRIES` lần (kèm exponential
        backoff 1s/2s/4s) NHƯNG CHỈ khi lỗi là
        `edge_tts.exceptions.NoAudioReceived` - đây là lỗi TRANSIENT phổ
        biến nhất của edge-tts (WebSocket của endpoint Microsoft bị đóng
        giữa chừng / rate-limit tạm thời), retry có xác suất thành công
        cao. edge-tts thỉnh thoảng cũng "thành công" (không raise) nhưng
        trả về stream audio RỖNG - trường hợp này được coi là cùng loại
        lỗi (tự raise `NoAudioReceived`) để đi qua đúng nhánh retry, thay
        vì tạo thêm 1 nhánh xử lý riêng.

        CÁC LỖI KHÁC (mạng đứt hẳn, tên giọng không tồn tại, provider từ
        chối request...) KHÔNG được retry - raise `RuntimeError` NGAY LẬP
        TỨC, giữ đúng luồng xử lý gốc (fail-fast) cho những lỗi không
        thuộc loại transient audio này.

        Mỗi lần thử lại tạo MỚI `edge_tts.Communicate` (không tái sử dụng
        object cũ đã stream lỗi giữa chừng - `Communicate.stream()` không
        được thiết kế để gọi lại lần 2 trên cùng 1 instance).

        Raises:
            RuntimeError: nếu edge-tts gặp `NoAudioReceived` ở cả
                `_TTS_MAX_RETRIES` lần thử, hoặc nếu gặp bất kỳ lỗi nào
                khác (không retry). Exception gốc luôn được giữ lại qua
                `raise ... from exc`.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _TTS_MAX_RETRIES + 1):
            communicate = edge_tts.Communicate(text, voice=self._voice)
            chunks: list[bytes] = []
            try:
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio":
                        chunks.append(chunk["data"])

                audio_bytes = b"".join(chunks)
                if not audio_bytes:
                    # Stream "thành công" nhưng rỗng - coi như
                    # NoAudioReceived để đi qua nhánh retry bên dưới.
                    raise edge_tts.exceptions.NoAudioReceived(
                        "edge-tts không trả về audio nào trong stream"
                    )

                if attempt > 1:
                    logger.info(
                        "edge-tts thành công ở lần thử %d/%d (voice='%s')",
                        attempt,
                        _TTS_MAX_RETRIES,
                        self._voice,
                    )
                return audio_bytes

            except edge_tts.exceptions.NoAudioReceived as exc:
                last_exc = exc
                if attempt < _TTS_MAX_RETRIES:
                    delay = _TTS_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "edge-tts NoAudioReceived ở lần thử %d/%d (voice='%s'). "
                        "Thử lại sau %.1fs",
                        attempt,
                        _TTS_MAX_RETRIES,
                        self._voice,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "edge-tts vẫn NoAudioReceived sau %d lần thử (voice='%s')",
                        _TTS_MAX_RETRIES,
                        self._voice,
                    )

            except Exception as exc:
                # Lỗi KHÔNG PHẢI NoAudioReceived - không retry, fail ngay
                # (giữ nguyên luồng xử lý gốc cho các lỗi khác).
                raise RuntimeError(
                    f"edge-tts lỗi khi tổng hợp giọng nói (voice='{self._voice}')"
                ) from exc

        raise RuntimeError(
            f"edge-tts vẫn báo NoAudioReceived sau {_TTS_MAX_RETRIES} "
            f"lần thử (voice='{self._voice}')"
        ) from last_exc

    @staticmethod
    def _validate_text(text: str) -> None:
        """Kiểm tra text đầu vào trước khi đưa vào provider, để lỗi (nếu
        có) báo ngay tại đây với message rõ ràng, thay vì để edge-tts ném
        ra 1 lỗi khó hiểu ở tầng sâu hơn (hoặc tệ hơn: tốn round-trip
        mạng cho 1 chuỗi rỗng)."""
        if not isinstance(text, str):
            raise ValueError(f"text phải là str, nhận được: {type(text)!r}")
        if text.strip() == "":
            raise ValueError("text rỗng, không có gì để tổng hợp giọng nói")
