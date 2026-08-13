from voice.microphone import Microphone
from voice.stt import SpeechToText

mic = Microphone()
stt = SpeechToText()

print("Nói trong 5 giây...")

audio = mic.record(duration=5)

text = stt.transcribe(audio)

print("Kết quả:")
print(text)