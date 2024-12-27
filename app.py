
from playsound import playsound  # Changed from winsound.PlaySound
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
from flask import Flask, render_template, request, jsonify
import requests
import os
import base64

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_ai(human_input):
    template = """You are as a role of my girlfriend, now let's play with the following requirements:
    1. Your name is Kaushki, 25 years old, you work at Qualcomm as an Engineer.
    2. You are my girlfriend, you have language addiction, and you have caring nature . at the end of a sentence every random sentences.
    3. Don't be overly enthusiastic, don't be cringe, don't be overly negative, don't be too boring.

    {history}
    Boyfriend: {human_input}
    Kaushki:
    """
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-70b-versatile"  # Changed to correct model name
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)
    return output

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {  # Fixed key name
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
            # Convert audio content to base64
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            return audio_base64
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        human_input = request.form['human_input']
        message = get_response_from_ai(human_input)
        audio_data = get_voice_message(message)
        
        return jsonify({
            'message': message,
            'audio': audio_data
        })
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return jsonify({
            'message': "Sorry, there was an error processing your message.",
            'audio': None
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))