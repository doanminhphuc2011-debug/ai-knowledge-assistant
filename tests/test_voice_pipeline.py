"""
tests/test_voice_pipeline.py
Integration test cho orchestration của VoiceChat (voice/voice_chat.py).

CHỈ test THỨ TỰ GỌI và XỬ LÝ LỖI của pipeline:

    Microphone.record() -> SpeechToText.transcribe()
        -> TextNormalizer.normalize() -> chatbot.ask()
        -> TextToSpeech.synthesize() -> Speaker.play()

KHÔNG test hành vi thật của bất kỳ thành phần nào:
- KHÔNG dùng microphone thật.
- KHÔNG dùng loa thật.
- KHÔNG dùng faster-whisper thật.
- KHÔNG dùng edge-tts thật.
- KHÔNG dùng LLM thật.

Toàn bộ dependency của `VoiceChat` được thay bằng fake object, tận dụng
đúng cơ chế dependency injection có sẵn trong `VoiceChat.__init__`
(Phase 6): `microphone`/`stt`/`normalizer`/`tts`/`speaker` đều là tham
số optional. Riêng `chatbot.ask` được import trực tiếp dưới dạng hàm
(không phải object có thể inject) nên được patch qua
`unittest.mock.patch("voice.voice_chat.ask", ...)`.

Dùng `unittest` + `unittest.mock` - KHÔNG dùng `pytest`, theo đúng yêu
cầu của Phase 7A.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from voice.voice_chat import VoiceChat

# FAKE CLASSES
# 
# Mỗi fake chỉ implement đúng 1 method public mà VoiceChat thực sự gọi
# (record/transcribe/normalize/synthesize/play), không kế thừa từ class
# thật (Microphone/SpeechToText/...) - VoiceChat chỉ cần đúng "shape"
# API (duck typing), không cần đúng type, nên fake có thể độc lập hoàn
# toàn với voice/microphone.py, voice/stt.py, voice/tts.py, voice/
# speaker.py thật (không import bất kỳ file nào trong số đó).
#
# Mỗi fake nhận 1 `call_log: list[str]` DÙNG CHUNG giữa các fake trong
# cùng 1 test, và append tên bước của chính nó vào đó khi được gọi. Nhờ
# vậy test có thể verify ĐÚNG THỨ TỰ gọi giữa các bước (không chỉ verify
# "từng bước có được gọi hay không" một cách rời rạc).


class FakeMicrophone:
    """Giả lập Microphone.record() -> numpy array giả, không cần
    microphone thật."""

    def __init__(self, call_log: list, audio: np.ndarray | None = None) -> None:
        self._call_log = call_log
        self._audio = audio if audio is not None else np.zeros(4, dtype=np.float32)
        self.record_calls: list = []

    def record(self, duration: float) -> np.ndarray:
        self.record_calls.append(duration)
        self._call_log.append("record")
        return self._audio


class FakeSTT:
    """Giả lập SpeechToText.transcribe() -> "xin chào", không cần
    faster-whisper thật. Có thể cấu hình ném exception để test lỗi."""

    def __init__(
        self,
        call_log: list,
        text: str = "xin chào",
        exception: Exception | None = None,
    ) -> None:
        self._call_log = call_log
        self._text = text
        self._exception = exception
        self.transcribe_calls: list = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribe_calls.append(audio)
        self._call_log.append("transcribe")
        if self._exception is not None:
            raise self._exception
        return self._text


class FakeNormalizer:
    """Giả lập TextNormalizer.normalize() -> "xin chào"."""

    def __init__(self, call_log: list, text: str = "xin chào") -> None:
        self._call_log = call_log
        self._text = text
        self.normalize_calls: list = []

    def normalize(self, text: str) -> str:
        self.normalize_calls.append(text)
        self._call_log.append("normalize")
        return self._text


class FakeTTS:
    """Giả lập TextToSpeech.synthesize() -> b"audio", không cần
    edge-tts thật. Có thể cấu hình ném exception để test lỗi."""

    def __init__(
        self,
        call_log: list,
        audio: bytes = b"audio",
        exception: Exception | None = None,
    ) -> None:
        self._call_log = call_log
        self._audio = audio
        self._exception = exception
        self.synthesize_calls: list = []

    def synthesize(self, text: str) -> bytes:
        self.synthesize_calls.append(text)
        self._call_log.append("synthesize")
        if self._exception is not None:
            raise self._exception
        return self._audio


class FakeSpeaker:
    """Giả lập Speaker.play() - không phát loa thật, chỉ lưu trạng thái
    đã được gọi. Có thể cấu hình ném exception để test lỗi."""

    def __init__(self, call_log: list, exception: Exception | None = None) -> None:
        self._call_log = call_log
        self._exception = exception
        self.play_calls: list = []

    def play(self, audio: bytes) -> None:
        self.play_calls.append(audio)
        self._call_log.append("play")
        if self._exception is not None:
            raise self._exception

# TEST CASES
class TestVoiceChatPipeline(unittest.TestCase):
    """Verify VoiceChat.listen_once() nối đúng thứ tự 6 bước, truyền
    đúng dữ liệu giữa các bước, và raise đúng RuntimeError khi 1 bước
    bất kỳ lỗi - không đụng tới bất kỳ thành phần thật nào."""

    def setUp(self) -> None:
        self.call_log: list = []
        self.fake_microphone = FakeMicrophone(self.call_log)
        self.fake_stt = FakeSTT(self.call_log)
        self.fake_normalizer = FakeNormalizer(self.call_log)
        self.fake_tts = FakeTTS(self.call_log)
        self.fake_speaker = FakeSpeaker(self.call_log)

    def _make_voice_chat(self) -> VoiceChat:
        """Khởi tạo VoiceChat với toàn bộ 5 dependency là fake, qua
        đúng cơ chế dependency injection của VoiceChat.__init__ - không
        instance nào trong số Microphone/SpeechToText/TextNormalizer/
        TextToSpeech/Speaker thật được tạo ra."""
        return VoiceChat(
            microphone=self.fake_microphone,
            stt=self.fake_stt,
            normalizer=self.fake_normalizer,
            tts=self.fake_tts,
            speaker=self.fake_speaker,
        )

    # TEST CASE 1
    def test_full_pipeline_calls_each_step_in_order_and_returns_answer(self) -> None:
        """record -> transcribe -> normalize -> ask -> synthesize ->
        play, đúng thứ tự; listen_once() trả về đúng câu trả lời của
        chatbot."""

        def fake_ask(text: str) -> str:
            # Verify normalize() đã chạy TRƯỚC ask() và đúng giá trị nó
            # trả ra được truyền tiếp cho ask().
            self.assertEqual(text, "xin chào")
            self.call_log.append("ask")
            return "Xin chào từ chatbot"

        with patch("voice.voice_chat.ask", side_effect=fake_ask) as mock_ask:
            voice_chat = self._make_voice_chat()
            result = voice_chat.listen_once(duration=3.0)

        self.assertEqual(result, "Xin chào từ chatbot")
        self.assertEqual(
            self.call_log,
            ["record", "transcribe", "normalize", "ask", "synthesize", "play"],
        )

        # Verify dữ liệu được truyền đúng giữa các bước (output bước
        # trước = input bước sau).
        self.assertEqual(self.fake_microphone.record_calls, [3.0])
        self.assertEqual(len(self.fake_stt.transcribe_calls), 1)
        self.assertEqual(self.fake_normalizer.normalize_calls, ["xin chào"])
        mock_ask.assert_called_once_with("xin chào")
        self.assertEqual(self.fake_tts.synthesize_calls, ["Xin chào từ chatbot"])
        self.assertEqual(self.fake_speaker.play_calls, [b"audio"])

    # TEST CASE 2
    def test_stt_error_raises_runtime_error_mentioning_stt(self) -> None:
        """SpeechToText.transcribe() ném exception -> listen_once()
        raise RuntimeError, message chứa 'STT'. Các bước sau (chatbot,
        TTS, Speaker) KHÔNG được gọi."""
        self.fake_stt = FakeSTT(self.call_log, exception=ValueError("stt boom"))

        with patch("voice.voice_chat.ask") as mock_ask:
            voice_chat = self._make_voice_chat()
            with self.assertRaises(RuntimeError) as cm:
                voice_chat.listen_once(duration=3.0)

        self.assertIn("STT", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, ValueError)  # không nuốt exception gốc
        mock_ask.assert_not_called()
        self.assertEqual(self.fake_tts.synthesize_calls, [])
        self.assertEqual(self.fake_speaker.play_calls, [])
        self.assertEqual(self.call_log, ["record", "transcribe"])

    # TEST CASE 3
    def test_chatbot_ask_error_raises_runtime_error_mentioning_chatbot(self) -> None:
        """chatbot.ask() ném exception -> listen_once() raise
        RuntimeError, message chứa 'chatbot'. Các bước sau (TTS,
        Speaker) KHÔNG được gọi."""
        with patch("voice.voice_chat.ask", side_effect=RuntimeError("llm boom")):
            voice_chat = self._make_voice_chat()
            with self.assertRaises(RuntimeError) as cm:
                voice_chat.listen_once(duration=3.0)

        self.assertIn("chatbot", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertEqual(self.fake_tts.synthesize_calls, [])
        self.assertEqual(self.fake_speaker.play_calls, [])
        self.assertEqual(self.call_log, ["record", "transcribe", "normalize"])

    # TEST CASE 4
    def test_tts_error_raises_runtime_error_mentioning_tts(self) -> None:
        """TextToSpeech.synthesize() ném exception -> listen_once()
        raise RuntimeError, message chứa 'TTS'. Speaker KHÔNG được
        gọi."""
        self.fake_tts = FakeTTS(self.call_log, exception=ValueError("tts boom"))

        with patch("voice.voice_chat.ask", return_value="Xin chào từ chatbot"):
            voice_chat = self._make_voice_chat()
            with self.assertRaises(RuntimeError) as cm:
                voice_chat.listen_once(duration=3.0)

        self.assertIn("TTS", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertEqual(self.fake_speaker.play_calls, [])
        self.assertEqual(
            self.call_log, ["record", "transcribe", "normalize", "synthesize"]
        )

    # TEST CASE 5
    def test_speaker_error_raises_runtime_error_mentioning_speaker(self) -> None:
        """Speaker.play() ném exception -> listen_once() raise
        RuntimeError, message chứa 'Speaker'."""
        self.fake_speaker = FakeSpeaker(self.call_log, exception=OSError("speaker boom"))

        with patch("voice.voice_chat.ask", return_value="Xin chào từ chatbot"):
            voice_chat = self._make_voice_chat()
            with self.assertRaises(RuntimeError) as cm:
                voice_chat.listen_once(duration=3.0)

        self.assertIn("Speaker", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, OSError)
        self.assertEqual(
            self.call_log,
            ["record", "transcribe", "normalize", "synthesize", "play"],
        )


if __name__ == "__main__":
    unittest.main()
