import whisper
from gtts import gTTS
import os
import gdown

# Install Whisper model from GitHub (if not already installed)
# You can remove the pip install line if you have already installed it in your environment.
# !pip install git+https://github.com/openai/whisper.git

# Load Whisper model
model = whisper.load_model('medium')

# Download audio file using gdown
gdown.download('https://drive.google.com/uc?id=12u4YgTFzi5NbClJ8M9ZzSuLnsqkI_bQA', '/content/flowers_12-14.wav', quiet=False)

# Transcribe audio
result = model.transcribe('/content/flowers_12-14.wav', fp16=False)
text = result['text']

# Text to speech conversion
if not isinstance(text, str) or not text.strip():
    raise ValueError("Transcribed text is not valid for TTS conversion.")

tts = gTTS(text, lang='en')
output_audio_path = '/content/drive/MyDrive/output_speech_vtoto.mp3'
tts.save(output_audio_path)
