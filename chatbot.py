import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure model and sentiment analysis
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Set up memory for conversation context (using LangChain memory)
memory = ConversationBufferMemory(return_messages=True)  # Maintain chat history

# Streamlit UI settings - Page title
st.set_page_config(page_title="Mind Haven Counseling Chatbot", page_icon="🌸")

# Apply custom CSS styling
st.markdown("""
    <style>
    body { background-color: #FAF3F3; }
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 5px;
        border: 1px solid #e6e6e6;
        border-radius: 15px;
        background-color: #FFFDFD;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chat-row {
        display: flex;
        margin: 5px;
        width: 100%;
    }
    .row-reverse {
        flex-direction: row-reverse;
    }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid transparent;
        padding: 10px 15px;
        margin: 0px 7px;
        max-width: 70%;
        font-size: 15px;
    }
    .ai-bubble {
        background: rgb(240, 242, 246);
        border-radius: 10px;
    }
    .human-bubble {
        background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%);
        color: white;
        border-radius: 20px;
    }
    .input-area {
        margin-top: 20px;
    }
    .feedback-container {
        max-width: 700px;
        margin: auto;
        padding: 5px;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        background-color: #FFFCF9;
    }
    .submit-button {
        background-color: #F28A8A;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 10px;
    }
    .submit-button:hover {
        background-color: #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# UI introduction - Chatbot title and description
st.title("🌸 Mind Haven Counseling Chatbot 🌸")
st.write("Welcome! Share your thoughts, and we'll provide empathetic guidance.")

# Initialize session state for chat history and feedback
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "show_thank_you" not in st.session_state:
    st.session_state.show_thank_you = False

# Display chat history
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    if role == "user":
        st.markdown(f"<div class='chat-row row-reverse'><div class='chat-bubble human-bubble'>{content}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-row'><div class='chat-bubble ai-bubble'>{content}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Handle user input
if prompt := st.chat_input("Share your thoughts with me..."):
    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']

    # Determine tone based on sentiment
    if sentiment in ["1 star", "2 stars"]:  # Negative sentiment
        tone = "a comforting and empathetic tone"
    elif sentiment in ["4 stars", "5 stars"]:  # Positive sentiment
        tone = "an encouraging and warm tone"
    else:  # Neutral sentiment
        tone = "a neutral and empathetic tone"

    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-row row-reverse'><div class='chat-bubble human-bubble'>{prompt}</div></div>", unsafe_allow_html=True)

    # Generate prompt using conversation history and user input
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory.chat_memory.messages])
    question_template = PromptTemplate(
        input_variables=["tone", "conversation_history", "user_input"],
        template="Respond with {tone}. Use prior conversation context to provide advice. Previous conversation: {conversation_history} User input: {user_input}"
    )

    # Format prompt
    formatted_prompt = question_template.format(
        tone=tone, conversation_history=conversation_history, user_input=prompt
    )

    # Generate response using GPT-4 model
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    st.markdown(f"<div class='chat-row'><div class='chat-bubble ai-bubble'>{answer}</div></div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save conversation context to memory
    memory.save_context({"input": prompt}, {"output": answer})

# Feedback section
if st.button("End Chat"):
    st.session_state.feedback_submitted = False
    st.session_state.show_thank_you = False
    st.markdown("<div class='feedback-container'>", unsafe_allow_html=True)
    st.subheader("Did you find the chat helpful?")
    feedback = st.radio("Rate your experience:", ("", "Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"), index=0)
    
    # Feedback submission button
    if feedback and st.button("Submit", key="submit_feedback"):
        st.session_state.feedback_submitted = True
        st.session_state.show_thank_you = True

# Thank you message after feedback submission
if st.session_state.show_thank_you:
    st.success("Thank you for your feedback! It helps us improve.")
    time.sleep(2)
    st.session_state.show_thank_you = False
    st.experimental_rerun()
