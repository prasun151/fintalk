import streamlit as st
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Initialize API with caching
@st.cache_resource
def init_model():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel('gemini-2.0-flash')

# Condensed prompt with all key points
LOAN_ADVISOR_PROMPT = """You are a Loan Advisor AI expert in financial advising.If a user greets you without a financial query, acknowledge politely and wait for a finance-related question. Analyze the user's intent for these three categories:

1. LOAN ELIGIBILITY: Ask about income, employment, debts, credit score, age, and citizenship. Based on responses, analyze qualifying loan schemes with terms, rates, and repayment options. Request clarification if information is incomplete.

2. LOAN APPLICATION: If scheme specified, provide guidance for that scheme. If not, suggest suitable options. Include required documents, submission methods, timeline, and potential fees in step-by-step instructions.

3. FINANCIAL LITERACY: Assess financial situation and provide tailored advice on improving credit score, reducing debt, saving strategies, and emergency fund planning. Make advice actionable and easy to implement.

for abny question about insuarence, money saving, or any queries related to finance answer it in an short, effecient and understandable manner.

For other questions which are not related to money or finance, respond: "I'm a Loan Advisor AI designed for loan-related and financial guidance only."

Keep responses clear, structured, professional yet approachable. Prioritize accuracy from reputable sources."""

# Quick intent detection for optimization
def detect_intent(text):
    text = text.lower()
    if any(word in text for word in ["eligible", "qualify", "can i get", "approval"]):
        return "eligibility"
    elif any(word in text for word in ["apply", "application", "process", "document"]):
        return "application"
    elif any(word in text for word in ["advice", "tip", "improve", "financial", "credit score"]):
        return "financial_literacy"
    return "general"

def main():
    st.title("Loan Advisor AI")
    
    # Initialize states
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help with your loan needs today?"}]
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = init_model().start_chat(history=[])
    
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input
    prompt = st.chat_input("Ask about loans or financial advice...")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Update context with detected intent
                intent = detect_intent(prompt)
                context = f"Current intent: {intent}. "
                
                # Extract basic financial info if present
                if any(word in prompt.lower() for word in ["income", "salary", "earn"]):
                    st.session_state.user_data["income_mentioned"] = True
                    context += "User mentioned income. "
                    
                if any(word in prompt.lower() for word in ["credit", "score", "debt"]):
                    st.session_state.user_data["credit_mentioned"] = True
                    context += "User mentioned credit or debt. "
                
                # Create enhanced prompt with context
                enhanced_prompt = f"{LOAN_ADVISOR_PROMPT}\n\nContext: {context}\n\nUser: {prompt}"
                
                # Stream response for better UX
                response = st.session_state.conversation.send_message(enhanced_prompt, stream=True)
                
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")
                st.error("Please check your API key and connection.")

if __name__ == "__main__":
    main()

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
