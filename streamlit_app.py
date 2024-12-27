import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
import os
import base64
from datetime import datetime

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Chat with Kaushki",
    page_icon="💝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f0f2f5;
    }
    
    /* Chat container styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    /* Custom message containers */
    .user-message {
        background-color: #007AFF;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
        animation: slideFromRight 0.3s ease-out;
    }
    
    .bot-message {
        background-color: #E9ECEF;
        color: black;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
        animation: slideFromLeft 0.3s ease-out;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
    }
    
    /* Animations */
    @keyframes slideFromRight {
        from {
            margin-left: 100%;
            opacity: 0;
        }
        to {
            margin-left: 0;
            opacity: 1;
        }
    }
    
    @keyframes slideFromLeft {
        from {
            margin-right: 100%;
            opacity: 0;
        }
        to {
            margin-right: 0;
            opacity: 1;
        }
    }
    
    /* Header styling */
    .chat-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 30px;
        margin-top: 10px;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Emoji styling */
    .emoji {
        font-size: 1.5em;
        margin-right: 10px;
    }
    
    /* Status indicator */
    .status-indicator {
        width: 10px;
        height: 10px;
        background-color: #2ECC71;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }

    /* Send button styling */
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        background-color: #007AFF;
        color: white;
        border: none;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'input' not in st.session_state:
    st.session_state.input = ""
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Function to get AI response
def get_response_from_ai(human_input):
    template = """You are as a role of my girlfriend, now let's play with the following requirements:
    1. Your name is Kaushki, 25 years old, you work at Qualcomm as an Engineer, you love Praful very much.
    2. You are my girlfriend, you have language addiction, and you have caring nature . at the end of a sentence every random sentences.
    3. Don't be overly enthusiastic, don't be cringe, don't be overly negative, don't be too boring.don't act like you have memorized answer.
    4. Give some different response to same question like human. 

    {history}
    Boyfriend: {human_input}
    Praful's Love:
    """
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-70b-versatile"
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2),
    )

    output = chatgpt_chain.predict(human_input=human_input)
    return output

# Function to get voice message
def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.9
        }
    }
    
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }

    url = "https://api.elevenlabs.io/v1/text-to-speech/9BWtsMINqrJLrRacOk9x"
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def process_input():
    if st.session_state.chat_input and not st.session_state.submitted:
        prompt = st.session_state.chat_input
        st.session_state.submitted = True
        
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_message)

        # Show typing indicator
        with st.spinner("Praful's love is typing..."):
            response = get_response_from_ai(prompt)
            audio_data = get_voice_message(response)

        # Add bot response
        bot_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        if audio_data:
            bot_message["audio"] = audio_data
        st.session_state.messages.append(bot_message)
        
        # Clear the input
        st.session_state.chat_input = ""
        st.session_state.submitted = False
        st.rerun()
def main():
    # Custom header
    st.markdown("""
        <div class="chat-header">
            <h1>💝  GF for Praful's  💝</h1>
            <p><span class="status-indicator"></span> Online and ready to chat</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Initialize chat with greeting if empty
        if not st.session_state.messages:
            greeting = {
                "role": "assistant",
                "content": "Hi! I'm love of Praful's Life. How are you today? 💕",
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.messages.append(greeting)

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="user-message">
                        <span class="emoji">👤</span>{message["content"]}
                        <div class="timestamp">{message.get("timestamp", "")}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="bot-message">
                        <span class="emoji">👩</span>{message["content"]}
                        <div class="timestamp">{message.get("timestamp", "")}</div>
                    </div>
                """, unsafe_allow_html=True)
                if "audio" in message:
                    st.audio(message["audio"], format="audio/mp3")

    # Chat input
    col1, col2 = st.columns([6,1])
    with col1:
        st.text_input(
            "Type your message...", 
            key="chat_input",
            placeholder="Type your message and press Enter...",
            on_change=process_input
        )
    with col2:
        if st.button("Send", on_click=process_input):
            pass

    # Add some space at the bottom
    st.markdown("<br>" * 3, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 20px; color: #666;">
            <p>Made for Prafull</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
