import streamlit as st

st.title("Main Page")

if st.button("Start Chatting"):
    st.switch_page("pages/main.py")
