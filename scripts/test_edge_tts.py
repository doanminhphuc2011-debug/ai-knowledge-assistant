# scripts/test_edge_tts.py

from voice.tts import TextToSpeech

tts = TextToSpeech()

audio = tts.synthesize("Xin chào, tôi là trợ lý AI.")

print(len(audio))

with open("test.mp3", "wb") as f:
    f.write(audio)

print("saved")