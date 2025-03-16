# Multilingual Conversational Loan App

## Overview
The **Multilingual Conversational Loan App** is an AI-powered application that helps users inquire about loans, check eligibility, and receive recommendations in multiple languages. Built using Python and Streamlit, the app provides a seamless, user-friendly experience for global users.

## Features
- Supports multiple languages for a diverse user base.
- AI-powered chatbot for loan inquiries.
- Secure authentication and user data protection.
- Intuitive UI/UX with responsive design.
- Integration with financial APIs for real-time loan information.

## Installation Guide

### Prerequisites
Ensure you have the following installed:
- [Python](https://www.python.org/) (v3.8 or later recommended)
- [Git](https://git-scm.com/)
- A virtual environment manager (e.g., `venv` or `conda`)

### Steps to Set Up
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Suraj-787/fintalk.git
   cd fintalk
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run home.py
   ```

## Configuration
- Set up environment variables in a `.env` file (if required).
- Update API endpoints in `config.py`.
- Modify language settings in `locales/`.

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your branch and submit a pull request.