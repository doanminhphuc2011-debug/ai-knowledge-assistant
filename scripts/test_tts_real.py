from voice.tts import TextToSpeech

tts = TextToSpeech()

audio = tts.synthesize("Xin chào Minh Phúc")

print(type(audio))
print(len(audio))