import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Cache the chatbot model to avoid reloading on every interaction
@st.cache_resource(show_spinner=False)
def load_chatbot_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Cache the QA model to avoid reloading
@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load models
chatbot_model, chatbot_tokenizer = load_chatbot_model()
qa_pipeline = load_qa_model()

# Function to generate a chatbot response
def chatbot_response(user_input, chat_history_ids=None, max_length=50, temperature=0.7):
    # Tokenize the input
    new_input_ids = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors='pt')
    
    # Append to the conversation history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    
    # Generate response
    chat_history_ids = chatbot_model.generate(bot_input_ids, max_length=max_length, temperature=temperature, pad_token_id=chatbot_tokenizer.eos_token_id)
    
    # Decode the response
    response = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Function to get an answer from the QA model
def qa_response(user_question, context):
    return qa_pipeline(question=user_question, context=context)['answer']

# Streamlit interface layout
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Smart Chatbot with QA</h1>", unsafe_allow_html=True)

# Sidebar with settings for chatbot response control
st.sidebar.title("Chatbot Settings")
max_length = st.sidebar.slider("Max length of response", min_value=20, max_value=100, step=10, value=50)
temperature = st.sidebar.slider("Response creativity (Temperature)", min_value=0.1, max_value=1.0, step=0.1, value=0.7)
use_qa = st.sidebar.checkbox("Answer factual questions", value=False)

# Chat history (stored in session state for keeping conversation between interactions)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = None

# Input for providing background context for the QA model
context_input = st.text_area("Provide background/context for QA (optional)", "")

# User input
user_input = st.text_input("Ask a question or say something...")

# Process user input
if user_input:
    if use_qa and context_input:
        # Use the QA model if enabled and context is provided
        response = qa_response(user_input, context_input)
        st.session_state['chat_history'] = None  # Clear chat history for QA
    else:
        # Use the chatbot model for conversation
        response, st.session_state['chat_history'] = chatbot_response(user_input, st.session_state['chat_history'], max_length, temperature)
    
    # Display the response
    st.write(f"**Chatbot**: {response}")

# Button to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state['chat_history'] = None
    st.write("**Chatbot**: Conversation has been cleared.")

# Add some style to the interface
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 1.2rem;
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 1.1rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        font-size: 1.1rem;
        border: none;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)
