import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000/query"

def send_question_to_backend(question: str) -> str:
    """
    Sends the user question to the FastAPI RAG endpoint and returns the answer.
    """
    payload = {"question": question}
    try:
        resp = requests.post(FASTAPI_URL, json=payload)
        resp.raise_for_status()
        answer = resp.json().get("answer", "")
        return answer
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with backend: {e}")
        return "Sorry, couldn't get a response from the backend."

def query_llm():
    last_message = st.session_state["messages"][-1]
    if last_message["role"] == "user":
        user_question = last_message["content"]
        yield send_question_to_backend(user_question)

st.title("Local RAG - MLOps Lab 4")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(query_llm())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})