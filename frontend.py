import streamlit as st
from langchain_app import initialize_bot

# Initialize the chatbot
chat_function = initialize_bot()

@st.cache_resource
def get_chat_function():
    return chat_function

def get_response(prompt):
    chat = get_chat_function()
    try:
        with st.spinner("Thinking..."):
            response = chat(prompt)
            return response
    except Exception as e:
        return f"Error: {str(e)}"

st.title("IIIT Kottayam AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything about IIIT Kottayam"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    response = get_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
