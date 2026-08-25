"""
voice/tts.py
Text-to-Speech: Text -> Audio bytes.

THIẾT KẾ LẠI (fix vấn đề timeout 4s/8s + làm rõ bản chất "stream" của
edge-tts):

edge_tts.Communicate(...).stream() trả về NHIỀU FRAGMENT MP3 qua
websocket - đây KHÔNG PHẢI các file audio độc lập, ghép chúng lại
(`b"".join(...)`) mới ra 1 audio HOÀN CHỈNH decode được bằng soundfile.
Việc trước đây cố phát TỪNG fragment riêng (ở speaker.py) không phải
streaming thật, chỉ là gom-rồi-phát dưới lớp vỏ "stream" - đã bỏ hẳn cách
tiếp cận đó (xem voice_chat.py: kiến trúc mới stream Ở MỨC CÂU/ĐOẠN, TTS
mỗi câu tổng hợp thành audio HOÀN CHỈNH trước khi đưa cho Speaker).

TIMEOUT: bản cũ dùng 2 mức (4s cho chunk đầu, 8s cho các chunk sau) - quá
thấp, khiến edge-tts phản hồi hơi chậm (không phải lỗi thật) cũng bị coi
là treo, dẫn tới retry vô ích rồi fail hẳn. Giờ dùng ĐÚNG 1 mức timeout
rộng rãi (`_TTS_CHUNK_TIMEOUT_SECONDS`, mặc định 15s) áp dụng đồng nhất
cho MỌI lần chờ 1 fragment (kể cả fragment đầu tiên) - đủ lớn để chịu được
mạng chậm/server phản hồi trễ, nhưng vẫn có giới hạn để không treo vô hạn
khi kết nối THỰC SỰ chết.

API GIỮ NGUYÊN (backward-compatible):
    TextToSpeech.synthesize(text: str) -> bytes

API MỚI (không bắt buộc dùng, phục vụ pipeline sentence-level ở
voice_chat.py muốn tổng hợp câu kế tiếp trong lúc câu hiện tại đang chờ ở
1 coroutine khác thay vì 1 thread riêng):
    async def synthesize_async(text: str) -> bytes

`stream()` (AsyncIterator các fragment MP3 THÔ) vẫn được giữ lại để không
phá bất kỳ nơi gọi nào khác ngoài phạm vi 3 file được sửa lần này, NHƯNG
đã sửa lại timeout cho đúng bản chất ở trên, và docstring nói rõ: đây là
fragment MP3 thô, người gọi PHẢI tự gom đủ fragment của 1 câu trước khi
decode/phát - không được coi từng fragment là 1 audio độc lập.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator

import edge_tts
import edge_tts.exceptions

from voice.config import get_voice_config
from voice.tts_text_prep import prepare_tts_text

logger = logging.getLogger(__name__)

_TTS_VOICE_ENV_VAR = "VOICE_TTS_VOICE"

# RETRY: edge_tts.exceptions.NoAudioReceived thường là lỗi TRANSIENT
# (WebSocket bị đóng giữa chừng / rate-limit tạm thời) - retry có xác suất
# thành công cao. Vì mỗi lần gọi TTS giờ chỉ tổng hợp 1 CÂU NGẮN (kiến
# trúc sentence-level ở voice_chat.py), retry cả 1 câu ngắn là rẻ, an
# toàn - không phải retry cả câu trả lời dài.
_TTS_MAX_RETRIES = 3
_TTS_RETRY_BASE_DELAY_SECONDS = 1.0

# TIMEOUT DUY NHẤT cho mỗi lần chờ 1 fragment từ edge-tts (áp dụng đồng
# nhất cho fragment đầu tiên lẫn các fragment sau, KHÔNG còn phân biệt 2
# mức 4s/8s như bản cũ - phân biệt đó là nguyên nhân gây fail giả khi
# server chỉ đang phản hồi chậm, xem docstring đầu file). 15s đủ rộng rãi
# cho mạng chậm nhưng vẫn chặn được treo vô hạn khi kết nối thực sự chết.
_TTS_CHUNK_TIMEOUT_SECONDS = 15.0


class TextToSpeech:
    """Bọc `edge_tts.Communicate`, expose `synthesize()`/`synthesize_async()`
    (audio HOÀN CHỈNH của 1 đoạn text) và `stream()` (fragment MP3 thô,
    xem cảnh báo ở docstring đầu file)."""

    def __init__(self) -> None:
        config = get_voice_config()

        if not config.enabled:
            raise RuntimeError(
                "TextToSpeech được khởi tạo nhưng VOICE_ENABLED=False. "
                "Không nên khởi tạo TTS khi Voice đang tắt."
            )

        voice = os.getenv(_TTS_VOICE_ENV_VAR)
        if voice is None or voice.strip() == "":
            raise RuntimeError(
                f"Thiếu biến môi trường {_TTS_VOICE_ENV_VAR} trong .env."
            )
        self._voice = voice.strip()
        logger.info("Đã khởi tạo TextToSpeech (provider=edge-tts, voice='%s')", self._voice)

    def synthesize(self, text: str) -> bytes:
        """Chuyển text thành audio bytes HOÀN CHỈNH (API cũ, giữ nguyên
        cho backward compatibility). Chạy `synthesize_async()` trong 1
        event loop mới (`asyncio.run`) - an toàn gọi từ code đồng bộ hoặc
        từ 1 thread riêng (không phải thread đang chạy event loop khác)."""
        return asyncio.run(self.synthesize_async(text))

    async def synthesize_async(self, text: str) -> bytes:
        """Bản async của `synthesize()` - dùng khi caller đã ở trong 1
        coroutine/event loop sẵn có (tránh gọi `asyncio.run()` lồng nhau,
        vốn sẽ lỗi `RuntimeError: asyncio.run() cannot be called from a
        running event loop`)."""
        self._validate_text(text)

        prep_start = time.perf_counter()
        tts_text = prepare_tts_text(text)
        logger.info("[TTS] prepare completed in %.2fs", time.perf_counter() - prep_start)

        total_start = time.perf_counter()
        try:
            audio_bytes = await self._gather_audio(tts_text)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception(
                "Lỗi khi tổng hợp giọng nói (voice='%s', độ dài text=%d ký tự)",
                self._voice,
                len(tts_text),
            )
            raise RuntimeError("Không thể tổng hợp giọng nói (TTS)") from exc

        if not audio_bytes:
            logger.error("edge-tts không trả về audio nào cho voice='%s'", self._voice)
            raise RuntimeError("Provider edge-tts không trả về audio nào")

        logger.info("[TTS] provider completed in %.2fs", time.perf_counter() - total_start)
        logger.info("[TTS] audio bytes = %d", len(audio_bytes))
        return audio_bytes

    async def _gather_audio(self, tts_text: str) -> bytes:
        """Gọi edge-tts, GOM toàn bộ fragment MP3 của `tts_text` thành 1
        bytes duy nhất (audio HOÀN CHỈNH, decode được ngay bằng
        soundfile). Retry tối đa `_TTS_MAX_RETRIES` lần với backoff mũ,
        CHỈ khi lỗi là `NoAudioReceived`/timeout (nhận dở rồi mất kết nối
        thì coi là lỗi cứng của lần thử này, KHÔNG cố ghép nối phần dở -
        dễ tạo audio hỏng/lặp; retry lại từ đầu 1 câu NGẮN vẫn rẻ)."""
        last_exc: Exception | None = None

        for attempt in range(1, _TTS_MAX_RETRIES + 1):
            communicate = edge_tts.Communicate(tts_text, voice=self._voice)
            chunks: list[bytes] = []
            has_received_any = False
            try:
                stream_iter = communicate.stream()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=_TTS_CHUNK_TIMEOUT_SECONDS
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise edge_tts.exceptions.NoAudioReceived(
                            f"Timeout sau {_TTS_CHUNK_TIMEOUT_SECONDS}s không nhận "
                            "được fragment từ edge-tts"
                        )

                    if chunk.get("type") == "audio" and chunk.get("data"):
                        has_received_any = True
                        chunks.append(chunk["data"])

                audio_bytes = b"".join(chunks)
                if not audio_bytes:
                    raise edge_tts.exceptions.NoAudioReceived(
                        "edge-tts không trả về audio nào trong stream"
                    )

                if attempt > 1:
                    logger.info(
                        "edge-tts thành công ở lần thử %d/%d (voice='%s')",
                        attempt, _TTS_MAX_RETRIES, self._voice,
                    )
                return audio_bytes

            except edge_tts.exceptions.NoAudioReceived as exc:
                last_exc = exc
                if attempt < _TTS_MAX_RETRIES:
                    delay = _TTS_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "edge-tts NoAudioReceived ở lần thử %d/%d (voice='%s', "
                        "đã nhận fragment trước đó=%s). Thử lại sau %.1fs",
                        attempt, _TTS_MAX_RETRIES, self._voice, has_received_any, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "edge-tts vẫn NoAudioReceived sau %d lần thử (voice='%s')",
                        _TTS_MAX_RETRIES, self._voice,
                    )
            except Exception as exc:
                # Lỗi KHÔNG PHẢI NoAudioReceived/timeout - không retry, fail ngay.
                raise RuntimeError(
                    f"edge-tts lỗi khi tổng hợp giọng nói (voice='{self._voice}')"
                ) from exc

        raise RuntimeError(
            f"edge-tts vẫn báo NoAudioReceived sau {_TTS_MAX_RETRIES} lần thử"
        ) from last_exc

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield TRỰC TIẾP các fragment MP3 THÔ từ edge-tts, theo đúng
        timeout duy nhất `_TTS_CHUNK_TIMEOUT_SECONDS` (KHÔNG còn 2 mức
        4s/8s của bản cũ).

        CẢNH BÁO CHO NGƯỜI GỌI: mỗi fragment yield ra KHÔNG PHẢI 1 audio
        độc lập - đây là các mảnh MP3 chưa hoàn chỉnh. Muốn có audio decode
        được (bằng soundfile/`Speaker.play()`), phải tự gom hết fragment
        của 1 đoạn text rồi `b"".join(...)` trước, KHÔNG được đưa thẳng
        từng fragment cho `Speaker.play()`. Nếu chỉ cần audio hoàn chỉnh,
        dùng `synthesize()`/`synthesize_async()` thay vì hàm này - 2 hàm
        đó đã tự làm đúng việc gom fragment.

        Giữ lại hàm này chỉ để không phá code khác (ngoài phạm vi 3 file
        đang sửa) có thể đang gọi `TextToSpeech.stream()` trực tiếp - kiến
        trúc mới ở `voice_chat.py` KHÔNG dùng hàm này nữa."""
        self._validate_text(text)

        prep_start = time.perf_counter()
        tts_text = prepare_tts_text(text)
        logger.info("[TTS] prepare completed in %.2fs", time.perf_counter() - prep_start)

        last_exc: Exception | None = None
        for attempt in range(1, _TTS_MAX_RETRIES + 1):
            stream_start = time.perf_counter()
            communicate = edge_tts.Communicate(tts_text, voice=self._voice)
            has_received_any = False
            chunk_count = 0
            total_bytes = 0
            try:
                stream_iter = communicate.stream()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=_TTS_CHUNK_TIMEOUT_SECONDS
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise edge_tts.exceptions.NoAudioReceived(
                            f"Timeout sau {_TTS_CHUNK_TIMEOUT_SECONDS}s không nhận "
                            "được fragment từ edge-tts"
                        )

                    if chunk.get("type") == "audio" and chunk.get("data"):
                        has_received_any = True
                        chunk_count += 1
                        total_bytes += len(chunk["data"])
                        yield chunk["data"]

                if not has_received_any:
                    raise edge_tts.exceptions.NoAudioReceived(
                        "edge-tts không trả về audio nào trong stream"
                    )

                logger.info(
                    "[TTS] streaming completed in %.2fs | fragments = %d | bytes = %d",
                    time.perf_counter() - stream_start, chunk_count, total_bytes,
                )
                return

            except edge_tts.exceptions.NoAudioReceived as exc:
                last_exc = exc
                if has_received_any:
                    logger.warning("Mất kết nối giữa chừng stream ở lần thử %d", attempt)
                    break
                if attempt < _TTS_MAX_RETRIES:
                    delay = _TTS_RETRY_BASE_DELAY_SECONDS * (1.5 ** (attempt - 1))
                    logger.warning(
                        "edge-tts không phản hồi ở lần thử %d/%d (voice='%s'). "
                        "Retry sau %.1fs...",
                        attempt, _TTS_MAX_RETRIES, self._voice, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "edge-tts không thể kết nối sau %d lần thử", _TTS_MAX_RETRIES
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"edge-tts lỗi khi stream audio (voice='{self._voice}')"
                ) from exc

        raise RuntimeError(
            f"edge-tts vẫn báo NoAudioReceived sau {_TTS_MAX_RETRIES} lần thử"
        ) from last_exc

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise ValueError(f"text phải là str, nhận được: {type(text)!r}")
        if text.strip() == "":
            raise ValueError("text rỗng, không có gì để tổng hợp giọng nói")
