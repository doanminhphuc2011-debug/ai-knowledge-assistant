from voice.voice_chat import VoiceChat

chat = VoiceChat()

print("Bắt đầu hội thoại...")

response = chat.listen_once(duration=5)

print()
print("BOT:")
print(response)