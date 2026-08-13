from voice.microphone import Microphone

mic = Microphone()

print("Thu âm 5 giây...")

audio = mic.record(duration=5)

print(type(audio))
print(audio.shape)
print(audio.dtype)
print(audio[:10])