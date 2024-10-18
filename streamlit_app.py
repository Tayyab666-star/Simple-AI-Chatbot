# Step 1: Install necessary libraries
# You can run this in your environment, like Google Colab, local machine, or a cloud VM
!pip install transformers
!pip install streamlit

# Step 2: Import necessary libraries
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Step 3: Load the models using Streamlit's caching to improve response time
@st.cache_resource(show_spinner=False)
def load_chatbot_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load models
chatbot_model, chatbot_tokenizer = load_chatbot_model()
qa_pipeline = load_qa_model()

# Step 4: Create the chatbot response function
def chatbot_response(user_input, chat_history_ids=None, max_length=50, temperature=0.7):
    # Tokenize the user input
    new_input_ids = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors='pt')

    # Append user input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate response using the model
    chat_history_ids = chatbot_model.generate(bot_input_ids, max_length=max_length, temperature=temperature, pad_token_id=chatbot_tokenizer.eos_token_id)

    # Decode the response
    response = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

# Step 5: Create a function for the QA model
def qa_response(user_question, context):
    # Generate answer using the question-answering pipeline
    answer = qa_pipeline(question=user_question, context=context)
    return answer['answer']

# Step 6: Streamlit interface
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Smart Chatbot with QA</h1>", unsafe_allow_html=True)

# Sidebar configuration for user options
st.sidebar.title("Chatbot Settings")
max_length = st.sidebar.slider("Max length of response", min_value=20, max_value=100, step=10, value=50)
temperature = st.sidebar.slider("Response creativity (Temperature)", min_value=0.1, max_value=1.0, step=0.1, value=0.7)
use_qa = st.sidebar.checkbox("Answer factual questions", value=False)

# Chat history (stored as a session state to maintain state between interactions)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = None

# Context input for the QA model
context_input = st.text_area("Provide some background/context (for QA model)", "")

# User input
user_input = st.text_input("Type your message here...")

# If there's user input, process it
if user_input:
    if use_qa and context_input:
        # If QA mode is enabled and context is provided, use the QA model
        response = qa_response(user_input, context_input)
        st.session_state['chat_history'] = None  # Reset chat history when using QA model
    else:
        # Use the chatbot model for general conversation
        response, st.session_state['chat_history'] = chatbot_response(user_input, st.session_state['chat_history'], max_length, temperature)
    
    # Display the chatbot's response
    st.write(f"**Chatbot**: {response}")

# Display a button to clear chat history
if st.button("Clear Conversation"):
    st.session_state['chat_history'] = None
    st.write("**Chatbot**: Conversation has been cleared.")

# Add some style to the UI
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

