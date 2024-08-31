!pip install git+https://github.com/openai/whisper.git
import whisper
model = whisper.load_model('medium')
!gdown '12u4YgTFzi5NbClJ8M9ZzSuLnsqkI_bQA'
result = model.transcribe('/content/flowers_12-14.wav',fp16=False)
result['text']
text=result['text']
!pip install gtts
from gtts import gTTS
tts = gTTS(text, lang='en')
import os
if not isinstance(text, str) or not text.strip():
    raise ValueError("Transcribed text is not valid for TTS conversion.")

output_audio_path = '/content/drive/MyDrive/output_speech_vtoto.mp3'
tts.save(output_audio_path)
