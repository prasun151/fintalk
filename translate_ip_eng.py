from langdetect import detect
from dotenv import load_dotenv
import requests
import os

load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API key is missing!")

url = "https://api.sarvam.ai/translate"

headers = {
    "api-subscription-key": api_key,
    "Content-Type": "application/json"
}

def translate_to_english(text):
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "hi"  # Default to Hindi if detection fails

    lang_map = {
        "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
        "mr": "mr-IN", "kn": "kn-IN", "gu": "gu-IN", "ml": "ml-IN",
        "pa": "pa-IN", "ur": "ur-IN"
    }
    detected_lang_code = lang_map.get(detected_lang, "hi-IN")

    payload = {
        "source_language_code": detected_lang_code,
        "target_language_code": "en-IN",
        "speaker_gender": "Male",
        "mode": "classic-colloquial",
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "input": text
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        translated_text = response_json.get("translated_text", "Translation not available")
    else:
        translated_text = "[Translation Failed]"

    return (detected_lang_code, translated_text)  # Ensure tuple return
