import pyaudio
import wave
import numpy as np
import os
import requests
import base64
import time
import langid
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("API_KEY")

#Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample Rate
CHUNK = 1024
OUTPUT_FILE = "output.wav"

# Silence Detection Parameters
SILENCE_THRESHOLD = 500  
SILENCE_DURATION = 3

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open Stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)

print("Recording... Speak now!")

frames = []
silent_chunks = 0
max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)  

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

    # Convert audio data to numpy array for analysis
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # Compute volume level
    volume = np.abs(audio_data).mean()

    if volume < SILENCE_THRESHOLD:
        silent_chunks += 1
    else:
        silent_chunks = 0  

    if silent_chunks > max_silent_chunks:
        print("Silence detected. Stopping recording.")
        break

print("Finished Recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save audio file
waveFile = wave.open(OUTPUT_FILE, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

print(f"Audio recorded and saved as {OUTPUT_FILE}")


def split_audio(input_file, chunk_length=30):
    """Splits the audio into 30-second chunks."""
    audio = wave.open(input_file, 'rb')
    frame_rate = audio.getframerate()
    num_frames = audio.getnframes()
    total_duration = num_frames / frame_rate

    print(f"Total audio duration: {total_duration:.2f} seconds")

    chunks = []
    start_time = 0
    index = 1

    while start_time < total_duration:
        end_time = min(start_time + chunk_length, total_duration)
        
        # Read the chunk frames
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        audio.setpos(start_frame)
        frames = audio.readframes(end_frame - start_frame)
        
        # Save the chunk
        chunk_filename = f"chunk_{index}.wav"
        chunk_file = wave.open(chunk_filename, 'wb')
        chunk_file.setnchannels(audio.getnchannels())
        chunk_file.setsampwidth(audio.getsampwidth())
        chunk_file.setframerate(frame_rate)
        chunk_file.writeframes(frames)
        chunk_file.close()
        
        chunks.append(chunk_filename)
        start_time += chunk_length
        index += 1

    audio.close()
    return chunks


def audio_to_text(audio_file_path):
    """Sends an audio file to the API and returns the transcript."""
    url = "https://api.sarvam.ai/speech-to-text"
    
    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
        print(f"Error: {audio_file_path} not found or empty!")
        return ""

    files = {
        "file": (audio_file_path, open(audio_file_path, "rb"), "audio/wav")
    }
    
    data = {
        "model": "saarika:v2",
        "language_code": "unknown",
        "with_timestamps": "false",
        "with_diarization": "false",
        "num_speakers": "1"
    }
    
    headers = {
        "Accept": "application/json",
        "API-Subscription-Key": API
    }
    
    response = requests.post(url, files=files, data=data, headers=headers)
    
    if response.status_code == 200:
        try:
            json_response = response.json()
            transcript = json_response.get("transcript", "No transcript found")
            return transcript
        except ValueError:
            print("Error: Invalid JSON response")
            return ""
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ""


def process_long_audio(input_file):
    """Splits and transcribes long audio files."""
    chunks = split_audio(input_file)
    final_transcript = ""

    for chunk in chunks:
        print(f"Processing {chunk}...")
        transcript = audio_to_text(chunk)
        final_transcript += transcript + " "
        os.remove(chunk)  

    return final_transcript.strip()


def split_text(text, max_length=500):
    """Splits text into chunks of max 500 characters."""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
def text_to_speech(full_transcript):
    """Sends text to the TTS API in chunks and merges audio files."""
    url2 = "https://api.sarvam.ai/text-to-speech"
    text_chunks = split_text(full_transcript)
    audio_files = []

    for i, chunk in enumerate(text_chunks):
        payload = {
            "inputs": [chunk],
            "target_language_code": "en-IN",
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.2,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": False,
            "model": "bulbul:v1",
            "eng_interpolation_wt": 123,
            "override_triplets": {}
        }
        
        headers = {
            "api-subscription-key": API, 
            "Content-Type": "application/json"
        }
        
        response = requests.post(url2, json=payload, headers=headers)
        

        if response.status_code == 200:
            try:
                response_json = response.json()
                if "audios" in response_json and len(response_json["audios"]) > 0:
                    audio_data = base64.b64decode(response_json["audios"][0])  # ✅ Extract first audio file
                    audio_file_name = f"tts_chunk_{i+1}.wav"

                    with open(audio_file_name, "wb") as audio_file:
                        audio_file.write(audio_data)

                    if os.path.getsize(audio_file_name) > 0:
                        audio_files.append(audio_file_name)
                        print(f"✅ TTS Chunk {i+1} processed successfully!")
                    else:
                        print(f"⚠️ Warning: {audio_file_name} is empty. Skipping...")
                        os.remove(audio_file_name)
                else:
                    print(f"❌ Error: API response missing 'audios' key or empty for chunk {i+1}")
            except Exception as e:
                print(f"❌ Error decoding base64 audio for chunk {i+1}: {str(e)}")
        else:
            print(f"❌ Error processing TTS Chunk {i+1}: {response.text}")

        # 🕒 Add delay to prevent rate limiting
        time.sleep(1)

    return audio_files

def merge_audio(audio_files, output_filename="final_output.wav"):
    """Merges multiple audio chunks into a single file."""
    if not audio_files:
        print("No valid audio files to merge.")
        return

    try:
        output_audio = wave.open(output_filename, 'wb')
        
        first_file = wave.open(audio_files[0], 'rb')
        output_audio.setnchannels(first_file.getnchannels())
        output_audio.setsampwidth(first_file.getsampwidth())
        output_audio.setframerate(first_file.getframerate())

        for file in audio_files:
            with wave.open(file, 'rb') as audio_chunk:
                output_audio.writeframes(audio_chunk.readframes(audio_chunk.getnframes()))

        output_audio.close()
        print(f"Final audio saved as {output_filename}")

        # Remove temporary files
        for file in audio_files:
            os.remove(file)

    except wave.Error as e:
        print(f"Error merging audio files: {e}")

def translate_text(input_text, language):
    url3 = "https://api.sarvam.ai/translate"
    
    payload = {
        "input": input_text,
        "source_language_code": language + "-IN",
        "target_language_code": "en-IN",
        "speaker_gender": "Female",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "output_script": "roman",
        "numerals_format": "international"
    }
    
    headers = {"Content-Type": "application/json",
               "api-subscription-key": API}
    
    response = requests.post(url3, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("translated_text", "")
    else:
        return "Request failed"
    

def translate_text2(input_text, language):
    url3 = "https://api.sarvam.ai/translate"
    
    payload = {
        "input": input_text,
        "source_language_code": "en-IN",
        "target_language_code": language + "-IN",
        "speaker_gender": "Female",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "output_script": "roman",
        "numerals_format": "international"
    }
    
    headers = {"Content-Type": "application/json",
               "api-subscription-key": API}
    
    response = requests.post(url3, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("translated_text", "")
    else:
        return "Request failed"



def transliterate_text(input_text, language):
    url = "https://api.sarvam.ai/transliterate"
    
    payload = {
        "input": input_text,
        "source_language_code": language + "-IN",
        "target_language_code": "en-IN",
        "numerals_format": "international",
        "spoken_form_numerals_language": "native",
        "spoken_form": False
    }
    headers = {"Content-Type": "application/json",
               "api-subscription-key": API}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        return {
            "request_id": response_data.get("request_id", ""),
            "transliterated_text": response_data.get("transliterated_text", "")
        }
    else:
        return {"error": f"Request failed with status code {response.status_code}"}
    
    



# Process recorded audio
full_transcript = process_long_audio(OUTPUT_FILE)
print("\nFinal Transcription:\n", full_transcript)
language = langid.classify(full_transcript)[0] 
print(language)
if(language != 'en'):
    tran_eng = translate_text(full_transcript, language)
    print(f"tranlated text: {tran_eng}")

if(language != 'en'):
    out_tran = translate_text2(tran_eng, language)
    print(f"Out tranlated text: {out_tran}")
    transl_text = transliterate_text(out_tran, language)
# Convert text to speech

audio_files = text_to_speech(out_tran)
merge_audio(audio_files)
