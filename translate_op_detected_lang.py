from dotenv import load_dotenv
import requests
import time
import os
from translate_ip_eng import detected_lang, translation

load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API key is missing!")

url = "https://api.sarvam.ai/translate"

headers = {
    "api-subscription-key": api_key,
    "Content-Type": "application/json"
}

# Ensure detected_lang is in API-compatible format
lang_map = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "mr": "mr-IN",
    "kn": "kn-IN", "gu": "gu-IN", "ml": "ml-IN", "pa": "pa-IN", "ur": "ur-IN"
}

detected_lang = lang_map.get(detected_lang, "hi-IN")  # Default to Hindi if not mapped
print(f"Target Language: {detected_lang}")

payload = {
    "source_language_code": "en-IN",
    "target_language_code": detected_lang,
    "speaker_gender": "Male",
    "mode": "formal",
    "model": "mayura:v1",
    "enable_preprocessing": True,
    "input": translation.strip()
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    response_json = response.json()
    translated_text = response_json.get("translated_text", "Translation not available")
    print(translated_text)
else:
    print(f"[Translation Failed] - {response.text}")  # Show API error message for debugging

time.sleep(0.5)
