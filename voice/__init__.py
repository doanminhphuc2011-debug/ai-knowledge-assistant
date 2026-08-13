"""
voice/
Package chứa Voice Chat client - đứng SONG SONG với CLI text chat, cùng
gọi vào chatbot.ask() làm public API duy nhất của Chatbot Core.

Package này KHÔNG import ngược vào chatbot.py/memory.py/rag/... để tránh
circular import; ngược lại, voice_chat.py (ở phase sau) sẽ import
`from chatbot import ask` như một client bình thường.

Hiện tại (Phase 1) mới có config.py. Các phase sau sẽ bổ sung dần:
stt.py, text_normalizer.py, tts.py, latency.py, voice_chat.py.
"""
from voice.config import VoiceConfig, get_voice_config

__all__ = ["VoiceConfig", "get_voice_config"]
