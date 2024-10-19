import whisper
from pathlib import Path
import json


model = whisper.load_model('tiny')
path = Path("C:\Murali\sample_audio.mp3")

result = model.transcribe(str(path), language='en', verbose=True)


# Alternatively, if you just want to save plain text (not as a JSON structure):
with open('transcript.txt', "w") as file:
    file.write(result['text'])