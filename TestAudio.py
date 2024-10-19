import whisper
import requests
import speech_recognition as sr
from pathlib import Path
from deep_translator import GoogleTranslator
import time
from pydub import AudioSegment

# Function to download the audio file
def download_audio(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded audio file: {filename}")
    else:
        raise Exception(f"Failed to download audio file: {response.status_code}")

# URL of the audio file
audio_url = "https://www.moviesoundclips.net/movies1/darkknightrises/darkness.mp3"
audio_path = Path("darkness.mp3")
wav_audio_path = Path("darkness.wav")

# Download the audio file if it doesn't exist
if not audio_path.is_file():
    print("Downloading audio...")
    download_audio(audio_url, audio_path)

# Convert MP3 to WAV using pydub
print("Converting MP3 to WAV...")
audio = AudioSegment.from_mp3(audio_path)
audio.export(wav_audio_path, format="wav")
print(f"Converted to WAV: {wav_audio_path}")

# Initialize Whisper model
whisper_model = whisper.load_model('tiny')

# Whisper Transcription
start_time = time.time()
print("Transcribing audio with Whisper...")
whisper_result = whisper_model.transcribe(str(wav_audio_path), language='en', verbose=True)
whisper_transcription = whisper_result['text']
whisper_time = time.time() - start_time
print(f"Whisper Transcription: {whisper_transcription}")
print(f"Whisper Time: {whisper_time:.2f} seconds")

# Google Speech Recognition
recognizer = sr.Recognizer()
start_time = time.time()
print("Transcribing audio with Google Speech Recognition...")
with sr.AudioFile(str(wav_audio_path)) as source:
    audio_data = recognizer.record(source)  # Read the entire audio file
    google_transcription = recognizer.recognize_google(audio_data)
google_time = time.time() - start_time
print(f"Google Transcription: {google_transcription}")
print(f"Google Time: {google_time:.2f} seconds")

# Translate the transcriptions using Google Translator
print("Translating Whisper transcription to Spanish...")
whisper_translated = GoogleTranslator(source='en', target='es').translate(whisper_transcription)

print("Translating Google transcription to Spanish...")
google_translated = GoogleTranslator(source='en', target='es').translate(google_transcription)

# Save transcriptions and translations to text files
with open('whisper_transcript.txt', "w") as file:
    file.write(whisper_transcription)

with open('google_transcript.txt', "w") as file:
    file.write(google_transcription)

with open('whisper_translated.txt', "w") as file:
    file.write(whisper_translated)

with open('google_translated.txt', "w") as file:
    file.write(google_translated)

# Print results
print("\n--- Results ---")
print(f"Whisper Transcription: {whisper_transcription}")
print(f"Whisper Translation: {whisper_translated}")
print(f"Whisper Time: {whisper_time:.2f} seconds")

print(f"\nGoogle Transcription: {google_transcription}")
print(f"Google Translation: {google_translated}")
print(f"Google Time: {google_time:.2f} seconds")
