from voice.tts import TextToSpeech
from voice.speaker import Speaker

tts = TextToSpeech()
speaker = Speaker()

audio = tts.synthesize("Xin chào Minh Phúc")

speaker.play(audio)

print("DONE")