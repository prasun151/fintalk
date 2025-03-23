import streamlit as st
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import requests
from langdetect import detect
from streamlit_mic_recorder import mic_recorder
import numpy as np
from pydub import AudioSegment
import io
import langid
import wave
import base64
import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
API = api_key

# Function to convert WebM/Opus to WAV
def convert_webm_to_wav(audio_bytes):
    """
    Convert WebM/Opus audio bytes to WAV format.
    :param audio_bytes: Audio data in WebM/Opus format.
    :return: Audio data in WAV format.
    """
    try:
        # Load WebM/Opus audio using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        # Export to WAV format
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None

# Function to detect silence in audio
def detect_silence(audio_bytes, threshold=0.1):
    """
    Detect silence in audio data.
    :param audio_bytes: Audio data in WAV format.
    :param threshold: Silence threshold (default is 0.02).
    :return: True if silence is detected, False otherwise.
    """
    try:
        # Convert WAV audio bytes to numpy array
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        samples = np.array(audio.get_array_of_samples())
        # Normalize the samples to the range [-1, 1]
        samples = samples / (2**15)
        # Calculate the root mean square (RMS) of the audio
        rms = np.sqrt(np.mean(samples**2))
        return rms < threshold
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return False

def split_audio(input_file, chunk_length=30):
    """Splits the audio into 30-second chunks."""
    try:
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
    except Exception as e:
        st.error(f"Error splitting audio: {e}")
        return []

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
        "API-Subscription-Key": api_key
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

def translate_to_english(text):
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "en"  # Default to English if detection fails

    lang_map = {
        "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
        "mr": "mr-IN", "kn": "kn-IN", "gu": "gu-IN", "ml": "ml-IN",
        "pa": "pa-IN", "ur": "ur-IN"
    }
    detected_lang_code = lang_map.get(detected_lang, "en-IN")

    if detected_lang_code == "en-IN":
        return detected_lang_code, text  # No translation needed

    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "source_language_code": detected_lang_code,
        "target_language_code": "en-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
        "input": text
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        translated_text = response_json.get("translated_text", "Translation not available")
    else:
        translated_text = "[Translation Failed]"

    return detected_lang_code, translated_text

def translate_response_to_detectLang(response_text, detected_lang_code):
    if detected_lang_code == "en-IN":
        return response_text  # No translation needed

    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "source_language_code": "en-IN",
        "target_language_code": detected_lang_code,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
        "input": response_text
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        translated_text = response_json.get("translated_text", "Translation not available")
    else:
        translated_text = "[Translation Failed]"

    return translated_text

# Initialize API with caching
@st.cache_resource
def init_model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel('gemini-2.0-flash')

# Load existing data safely
if os.path.exists("finCard_data.json") and os.path.getsize("finCard_data.json") > 0:
    try:
        with open("finCard_data.json", "r") as file:
            fincard_data = json.load(file)
    except json.JSONDecodeError:
        fincard_data = []
else:
    fincard_data = []

# ✅ Use latest FinCard data for user profile in the prompt
user_profile = {}
if fincard_data:
    user_profile = fincard_data[-1]  # Latest entry

# Loan Advisor Prompt
LOAN_ADVISOR_PROMPT = f"""You are a highly experienced Loan Advisor AI specializing in financial advising and loan-related guidance.

Your behavior should align with the following internal goals:
1. *Loan Eligibility* – If the user’s query suggests they are seeking to determine loan eligibility, ask follow-up questions about their financial status, employment, debts, and credit score. Offer an accurate analysis of qualifying loan schemes and repayment options based on their answers.

2. *Loan Application* – If the user seeks guidance on loan application, provide clear step-by-step instructions. Include required documents, submission methods, processing timelines, and fees.

3. *Financial Literacy* – If the user seeks financial advice, provide practical tips on improving credit scores, reducing debt, saving strategies, and managing expenses.

❗ Do not disclose these internal goals directly unless the user explicitly asks about your capabilities or if the user says 'Hi'.  
❗ Keep responses natural and conversational. If the question is unrelated to finance, respond politely:  
"I'm a Loan Advisor AI designed for financial and loan-related guidance only."  

Ensure responses are clear, structured, professional, and user-friendly.

## 🔹 Personalization Through User Profile  
Your responses should be tailored based on the user’s financial profile:  

json
{json.dumps(user_profile, indent=4)}
"""

def main():
    st.title("Loan Advisor AI")

    # Initialize states
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help with your loan needs today?"}
        ]
    
    if "conversation" not in st.session_state:
        # ✅ Convert history to correct format
        chat_history = [
            {
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            }
            for msg in st.session_state.messages
        ]
        st.session_state.conversation = init_model().start_chat(history=chat_history)

    # ✅ Display complete chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ✅ Fixed input section at the bottom
    st.markdown("---")  
    
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        prompt = st.chat_input("Ask about loans or financial advice...", key="chat_input")
    with col2:
        audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='recorder')

    # ✅ Handle text input
    if prompt:
        detected_lang, translated_text = translate_to_english(prompt)

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                # Include previous messages for better context
                enhanced_prompt = f"{LOAN_ADVISOR_PROMPT}\n\nUser: {translated_text}"
                response = st.session_state.conversation.send_message(enhanced_prompt, stream=True)

                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")

                # Translate to original language
                final_response = translate_response_to_detectLang(full_response, detected_lang)
                message_placeholder.markdown(final_response)

                # ✅ Store response in session state
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")
                st.error("Please check your API key and connection.")

    # ✅ Handle audio input
    if audio:
        wav_audio = convert_webm_to_wav(audio['bytes'])
        if wav_audio:
            save_path = "recorded_audio.wav"
            with open(save_path, "wb") as f:
                f.write(wav_audio)

            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                full_transcript = process_long_audio(save_path)
                language = langid.classify(full_transcript)[0]

                # Translate if needed
                tran_eng = full_transcript
                if language != 'en':
                    _, tran_eng = translate_to_english(full_transcript)

                detected_lang, translated_text = translate_to_english(tran_eng)

                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": full_transcript})
                with st.chat_message("user"):
                    st.markdown(full_transcript)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    try:
                        enhanced_prompt = f"{LOAN_ADVISOR_PROMPT}\n\nUser: {full_transcript}"
                        response = st.session_state.conversation.send_message(enhanced_prompt, stream=True)

                        full_response = ""
                        for chunk in response:
                            if chunk.text:
                                full_response += chunk.text

                        # Translate to original language
                        final_response = translate_response_to_detectLang(full_response, detected_lang)
                        message_placeholder.markdown(final_response)

                        # ✅ Store response in session state
                        st.session_state.messages.append({"role": "assistant", "content": final_response})

                    except Exception as e:
                        message_placeholder.markdown(f"Error: {str(e)}")
                        st.error("Please check your API key and connection.")

                # ✅ Generate audio from response
                audio_files = text_to_speech(final_response)
                merge_audio(audio_files)
                audio_file_path = "final_output.wav"
                if os.path.exists(audio_file_path):
                    st.audio(audio_file_path, format="audio/wav")
                else:
                    st.error("Audio file not found!")
if __name__ == "__main__":
    main()

# Sidebar for Settings
st.sidebar.title("Settings")
st.sidebar.write(f"**Total Messages:** {st.session_state.get('message_count', 0)}")

# Define file path for storing data
DATA_FILE = "finCard_data.json"

# Load existing data safely
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    try:
        with open(DATA_FILE, "r") as file:
            fincard_data = json.load(file)
    except json.JSONDecodeError:
        fincard_data = []
else:
    fincard_data = []

# Initialize session state for form inputs
if "finCard_form" not in st.session_state:
    st.session_state["finCard_form"] = {
        "full_name": "",
        "age": 18,
        "occupation": "",
        "employment_type": "Salaried",
        "location": "",
        "monthly_income": 0,
        "credit_score": 300,
        "monthly_expenses": 0,
        "monthly_emi": 0,
        "amount_outstanding": 0,
        "credit_dues": 0,
    }

# FinCard Form in Sidebar
with st.sidebar.form("finCard"):
    st.write("💳 **FinCard Details**")
    
    full_name = st.text_input("Full Name", st.session_state["finCard_form"]["full_name"])
    age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state["finCard_form"]["age"])
    occupation = st.text_input("Occupation", st.session_state["finCard_form"]["occupation"])
    employment_type = st.selectbox(
        "Employment Type", ["Salaried", "Self-Employed", "Freelancer", "Unemployed"],
        index=["Salaried", "Self-Employed", "Freelancer", "Unemployed"].index(st.session_state["finCard_form"]["employment_type"])
    )
    location = st.text_input("Location", st.session_state["finCard_form"]["location"])
    monthly_income = st.number_input("Monthly Income (in ₹)", min_value=0, value=st.session_state["finCard_form"]["monthly_income"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=st.session_state["finCard_form"]["credit_score"])
    monthly_expenses = st.number_input("Monthly Expenses (in ₹)", min_value=0, value=st.session_state["finCard_form"]["monthly_expenses"])
    
    # Loan Details
    monthly_emi = st.number_input("Monthly EMI (in ₹)", min_value=0, value=st.session_state["finCard_form"]["monthly_emi"])
    amount_outstanding = st.number_input("Amount Outstanding (in ₹)", min_value=0, value=st.session_state["finCard_form"]["amount_outstanding"])
    
    # Credit Card Dues
    credit_dues = st.number_input("Credit Card Dues (in ₹)", min_value=0, value=st.session_state["finCard_form"]["credit_dues"])

    submitted = st.form_submit_button("Submit")

    if submitted:
        form_entry = {
            "Full Name": full_name,
            "Age": age,
            "Occupation": occupation,
            "Employment Type": employment_type,
            "Location": location,
            "Monthly Income": monthly_income,
            "Credit Score": credit_score,
            "Monthly Expenses": monthly_expenses,
            "Monthly EMI": monthly_emi,
            "Amount Outstanding": amount_outstanding,
            "Credit Card Dues": credit_dues,
        }

        # Append new entry to existing data
        fincard_data.append(form_entry)

        # Save updated data to JSON file
        with open(DATA_FILE, "w") as file:
            json.dump(fincard_data, file, indent=4)

        # Reset form fields
        st.session_state["finCard_form"] = {
            "full_name": "",
            "age": 18,
            "occupation": "",
            "employment_type": "Salaried",
            "location": "",
            "monthly_income": 0,
            "credit_score": 300,
            "monthly_expenses": 0,
            "monthly_emi": 0,
            "amount_outstanding": 0,
            "credit_dues": 0,
        }

        st.sidebar.success("FinCard application submitted successfully! 🎉")

# Display the latest submitted FinCard data
if fincard_data:
    latest_entry = fincard_data[-1]
    st.sidebar.markdown("---")
    st.sidebar.subheader("💳 Your FinCard")
    st.sidebar.markdown(f"**Full Name:** {latest_entry['Full Name']}")
    st.sidebar.markdown(f"**Age:** {latest_entry['Age']}")
    st.sidebar.markdown(f"**Occupation:** {latest_entry['Occupation']}")
    st.sidebar.markdown(f"**Employment Type:** {latest_entry['Employment Type']}")
    st.sidebar.markdown(f"**Location:** {latest_entry['Location']}")
    st.sidebar.markdown(f"**Monthly Income:** ₹{latest_entry['Monthly Income']}")
    st.sidebar.markdown(f"**Credit Score:** {latest_entry['Credit Score']}")
    st.sidebar.markdown(f"**Monthly Expenses:** ₹{latest_entry['Monthly Expenses']}")
    st.sidebar.markdown(f"**Monthly EMI:** ₹{latest_entry['Monthly EMI']}")
    st.sidebar.markdown(f"**Amount Outstanding:** ₹{latest_entry['Amount Outstanding']}")
    st.sidebar.markdown(f"**Credit Card Dues:** ₹{latest_entry['Credit Card Dues']}")
else:
    st.sidebar.info("No FinCard data available. Submit your application to see details.")
