import streamlit as st
import pandas as pd
from groq import Client
from dotenv import load_dotenv
import os
import json
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
    st.write("Smart loans, wise choices—chat with your AI financial mentor today!")
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
st.sidebar.write(f"**Total Messages:** {st.session_state.get('message_count', 0)}")

# Define file path for storing data
DATA_FILE = "finCard_data.json"

# Load existing data if the file exists
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as file:
        fincard_data = json.load(file)
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
    st.write("💳 **Your FinCard**")
    
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


# Load existing data if the file exists
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as file:
        fincard_data = json.load(file)
else:
    fincard_data = []

# Display the latest submitted FinCard data
if fincard_data:
    latest_entry = fincard_data[-1]
    st.sidebar.markdown("---")
    st.sidebar.subheader("💳 Your FinCard Summary")
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
