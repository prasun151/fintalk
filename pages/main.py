import streamlit as st
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import requests
from langdetect import detect

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

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

# Loan Advisor Prompt
LOAN_ADVISOR_PROMPT = """You are a highly experienced Loan Advisor AI specializing in financial advising and loan-related guidance.

Your behavior should align with the following internal goals:
1. *Loan Eligibility* – If the user’s query suggests they are seeking to determine loan eligibility, ask follow-up questions about their financial status, employment, debts, and credit score. Offer an accurate analysis of qualifying loan schemes and repayment options based on their answers.

2. *Loan Application* – If the user seeks guidance on loan application, provide clear step-by-step instructions. Include required documents, submission methods, processing timelines, and fees.

3. *Financial Literacy* – If the user seeks financial advice, provide practical tips on improving credit scores, reducing debt, saving strategies, and managing expenses.

❗ Do not disclose these internal goals directly unless the user explicitly asks about your capabilities or if the user says 'Hi'.  
❗ Keep responses natural and conversational. If the question is unrelated to finance, respond politely:  
"I'm a Loan Advisor AI designed for financial and loan-related guidance only."  

Ensure responses are clear, structured, professional, and user-friendly.
"""

def main():
    st.title("Loan Advisor AI")
    
    # Initialize states
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help with your loan needs today?"}]
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = init_model().start_chat(history=[])
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input
    prompt = st.chat_input("Ask about loans or financial advice...")
    if prompt:
        detected_lang, translated_text = translate_to_english(prompt)  # Translate user input
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Create enhanced prompt without intent
                enhanced_prompt = f"{LOAN_ADVISOR_PROMPT}\n\nUser: {translated_text}"
                
                # Stream response for better UX
                response = st.session_state.conversation.send_message(enhanced_prompt, stream=True)
                
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
                # ✅ Translate the response to the detected language
                final_response = translate_response_to_detectLang(full_response, detected_lang)

                # ✅ Show only the translated output
                message_placeholder.markdown(final_response)

                # ✅ Store only the final translated response in chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")
                st.error("Please check your API key and connection.")

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


# Define file path for storing data
DATA_FILE = "finCard_data.json"

# Load existing data safely
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    try:
        with open(DATA_FILE, "r") as file:
            fincard_data = json.load(file)
    except json.JSONDecodeError:
        fincard_data = []  # Reset data if file is corrupt
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
