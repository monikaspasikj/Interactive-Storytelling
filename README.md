
### Transkripcija.py
The transcription code looks solid but make sure to check the paths where files are downloaded or saved, especially in Docker. You might need to adjust paths for file handling within Docker:

```python
import whisper
from gtts import gTTS
import os
import gdown

# Load Whisper model
model = whisper.load_model('medium')

# Download audio file using gdown
gdown.download('https://drive.google.com/uc?id=12u4YgTFzi5NbClJ8M9ZzSuLnsqkI_bQA', 'flowers_12-14.wav', quiet=False)

# Transcribe audio
result = model.transcribe('flowers_12-14.wav', fp16=False)
text = result['text']

# Text to speech conversion
if not isinstance(text, str) or not text.strip():
    raise ValueError("Transcribed text is not valid for TTS conversion.")

# Save output audio file
output_audio_path = 'output_speech_vtoto.mp3'  # Adjust the path as needed
tts = gTTS(text, lang='en')
tts.save(output_audio_path)

print("Transcription and TTS conversion completed successfully.")