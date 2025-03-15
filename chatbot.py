import streamlit as st
from groq import Client
from dotenv import load_dotenv
import os
import requests
from translate_ip_eng import translate_to_english  # Import translation function

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("API_KEY")  # API Key for translation

if not GROQ_API_KEY or not SARVAM_API_KEY:
    raise ValueError("Missing API keys!")

client = Client(api_key=GROQ_API_KEY)

# Streamlit page configuration
st.set_page_config(page_title="FinTalk", page_icon="💬", layout="centered")

# Navigation state
if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = False

def show_chatbot():
    st.session_state["show_chat"] = True

# Home Page
if not st.session_state["show_chat"]:
    st.title("Welcome to FinTalk! 💬")
    st.write("Click the button below to start chatting with your AI assistant.")
    if st.button("Start Chatting"):
        show_chatbot()
    st.stop()

# Translation Function (English to Original Language)
def translate_from_english(text, target_lang):
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "source_language_code": "en-IN",
        "target_language_code": target_lang,
        "speaker_gender": "Male",
        "mode": "classic-colloquial",
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "input": text
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("translated_text", text)
    return "[Translation Failed]"

# Chatbot Response Function
def get_response(prompt, chat_history):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=chat_history,
        temperature=0.8
    )
    return response.choices[0].message.content

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
if "message_count" not in st.session_state:
    st.session_state["message_count"] = 0

# Chat UI
st.title("💬 FinTalk")
st.write("Talk to your AI assistant in real-time!")

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.chat_message("user").write(f"**You:** {content}")
        else:
            st.chat_message("assistant").write(f"**Bot:** {content}")

# Handle User Input
if user_input := st.chat_input("Type your message..."):
    detected_lang, translated_input = translate_to_english(user_input)  # Translate to English
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(f"**You:** {user_input} (Translated: {translated_input})")

    bot_response = get_response(translated_input, st.session_state["messages"])  # AI response
    translated_response = translate_from_english(bot_response, detected_lang)  # Translate back to original language

    st.session_state.messages.append({"role": "assistant", "content": translated_response})
    st.chat_message("assistant").write(f"**Bot:** {translated_response} (Original: {bot_response})")

    st.session_state["message_count"] += 1

st.sidebar.title("Settings")
st.sidebar.write(f"**Total Messages:** {st.session_state['message_count']}")
