import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Define model
model_name = "gpt-4o-mini"
llm = ChatOpenAI(model_name=model_name, api_key=api_key)

# Prompt template 
template = """
You are an expert in answering questions about the content of the provided financial documents.
Analyze the given financial records and provide concise and accurate answers to the user's questions.

Here are some relevant financial records: {record}
Here is the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


chain = prompt | llm

# Streamlit config
st.set_page_config(page_title="💰 Financial Chatbot", layout="centered") # Updated title
st.title("📊 Financial Data Chatbot") # Updated title

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# User input
user_input = st.chat_input("Ask me about your financial data...") 

if user_input:
    # Show user's message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The retriever from vector.py is now configured for financial data
            record = retriever.invoke(user_input)
            response = chain.invoke({"record": record, "question": user_input}) 
            st.markdown(response.content)
            # Save assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})