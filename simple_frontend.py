import streamlit as st
from langchain_app import initialize_bot
import logging
import os


# Initialize the chatbot
chat_function = initialize_bot()

@st.cache_resource
def get_chat_function():
    return chat_function


# This function gets response from the backend;

def get_response(prompt):
    chat = get_chat_function()
    try:
        result = chat(prompt)
        return result["final_response"]
    except Exception as e:
        logging.error(f"Frontend error: {str(e)}")

st.title("Mr.G (IIITK AI Assistant) - Simple Version")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
# Explain this as the flow starts here;
if prompt := st.chat_input("Ask me anything about IIIT Kottayam"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        # This function gets response from the backend
        response = get_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

