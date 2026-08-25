"""
voice_text_main.py
Test Voice Chat bằng TEXT thay cho Microphone + STT.

TEXT -> Normalize -> chatbot.ask() -> sentence-level TTS -> Speaker.play()

File này chỉ dùng để test trong môi trường yên lặng.
Không thay đổi voice_main.py và không dùng microphone/STT.
Không gọi tts.stream() hoặc speaker.play_stream().
"""
from __future__ import annotations

import logging
import time

from chatbot import ask
from voice.speaker import Speaker
from voice.text_normalizer import TextNormalizer
from voice.tts import TextToSpeech
from voice.voice_chat import VoiceChat

logging.basicConfig(level=logging.INFO)


class _DummyMicrophone:
    pass


class _DummySTT:
    pass


def main() -> None:
    print("=" * 64)
    print("VOICE TEXT TEST - SENTENCE-LEVEL SEQUENTIAL TTS")
    print("Nhập TEXT thay cho Microphone + STT")
    print()
    print("Luồng:")
    print("Text -> Normalize -> Chatbot/NER -> Tool/LLM")
    print("     -> split câu -> TTS từng chunk -> Speaker.play() tuần tự")
    print()
    print("Không dùng tts.stream() / speaker.play_stream()")
    print("Nhập 'exit' để thoát")
    print("=" * 64)

    normalizer = TextNormalizer()
    tts = TextToSpeech()
    speaker = Speaker()

    # Dùng VoiceChat để tái sử dụng đúng _speak_answer() của
    # pipeline sentence-level mới, nhưng không khởi tạo mic/STT thật.
    voice_chat = VoiceChat(
        microphone=_DummyMicrophone(),
        stt=_DummySTT(),
        normalizer=normalizer,
        tts=tts,
        speaker=speaker,
    )

    while True:
        try:
            text = input("\n👤 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nThoát.")
            break

        if text.lower() == "exit":
            print("Thoát.")
            break

        if not text:
            continue

        pipeline_start = time.perf_counter()

        try:
            print("\n" + "-" * 64)
            print("Đang xử lý...")

            normalize_start = time.perf_counter()
            normalized_text = normalizer.normalize(text)
            normalize_time = time.perf_counter() - normalize_start

            print(f"✓ Normalize: {normalize_time:.2f}s")
            print(f"✓ Text sau normalize: {normalized_text}")

            if not normalized_text.strip():
                print("⚠ Text rỗng sau normalize.")
                continue

            # Chatbot thật: NER/PhoBERT -> Tool nếu đủ thông tin,
            # nếu chưa đủ thì LLM xử lý.
            chatbot_start = time.perf_counter()
            answer = ask(normalized_text)
            chatbot_time = time.perf_counter() - chatbot_start

            print(f"✓ Chatbot: {chatbot_time:.2f}s")
            print()
            print(f"🤖 Ori: {answer}")

            # Đây là pipeline TTS mới.
            # Không gọi stream() / play_stream().
            print("\n🔊 Bắt đầu sentence-level TTS...")

            tts_start = time.perf_counter()
            voice_chat._speak_answer(answer)
            tts_time = time.perf_counter() - tts_start

            total_time = time.perf_counter() - pipeline_start

            print(f"✓ TTS + Speaker: {tts_time:.2f}s")
            print(f"✓ Pipeline tổng: {total_time:.2f}s")
            print("-" * 64)

        except KeyboardInterrupt:
            print("\nDừng lượt hiện tại.")
            break
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "Lỗi khi chạy text voice test"
            )
            print(f"❌ Lỗi: {exc}")
            print("-" * 64)


if __name__ == "__main__":
    main()
