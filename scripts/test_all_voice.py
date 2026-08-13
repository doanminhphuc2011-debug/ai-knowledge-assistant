from voice.microphone import Microphone
from voice.stt import SpeechToText
from voice.text_normalizer import TextNormalizer
from voice.tts import TextToSpeech
from voice.speaker import Speaker

print("Testing Microphone...")
mic = Microphone()

print("Testing STT...")
stt = SpeechToText()

print("Testing Normalizer...")
normalizer = TextNormalizer()

print("Testing TTS...")
tts = TextToSpeech()

print("Testing Speaker...")
speaker = Speaker()

audio = tts.synthesize("Voice system test")

speaker.play(audio)

print("ALL OK")