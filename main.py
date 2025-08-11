import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
import threading
import os
from datetime import datetime
import pyttsx3
from dotenv import load_dotenv

load_dotenv()
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORDINGS_DIR = "recordings"

os.makedirs(RECORDINGS_DIR, exist_ok=True)

def generate_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

class SpeechRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        self.stream = None
        self.filename = None

    def start_recording(self):
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=FRAMES_PER_BUFFER)
        self.frames = []
        self.recording = True
        self.filename = generate_filename()
        while self.recording:
            data = self.stream.read(FRAMES_PER_BUFFER)
            self.frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        return self.filename

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError:
        return "❌ API error."

# def query_llm(prompt):
#     return f" You said '{prompt}'"
import requests


# 🔐 Paste your HF token here


# --- HF Model Config ---
HF_TOKEN = os.getenv("HF_TOKEN") 
def query_llm(prompt):
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]['generated_text']
        elif "error" in result:
            return f"❌ Model error: {result['error']}"
        else:
            return "❌ No valid response from LLM."
    except Exception as e:
        return f"❌ Failed to connect to LLM: {e}"

# Streamlit UI
st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")
st.title("🎙️ Voice-Controlled Chatbot")
st.caption("Speak, record, and view answers with audio playback.")

# Session state
if "thread" not in st.session_state:
    st.session_state.thread = None
if "active_recorder" not in st.session_state:
    st.session_state.active_recorder = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_qa" not in st.session_state:
    st.session_state.latest_qa = None

# 💬 Full chat history (oldest to newest)
st.subheader("💬 Conversation History")
if st.session_state.chat_history:
    for i, (q, r, path) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**Response:** {r}")
        st.audio(path, format="audio/wav")
        st.markdown("---")
else:
    st.info("No previous conversations yet.")

# 🎤 Recording controls
st.subheader("🎤 Record & Ask")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording"):
        st.session_state.active_recorder = SpeechRecorder()
        st.session_state.thread = threading.Thread(
            target=st.session_state.active_recorder.start_recording)
        st.session_state.thread.start()
        st.info("🎙️ Recording... Press Stop when done.")

with col2:
    if st.button("⏹️ Stop & Ask"):
        if st.session_state.active_recorder and st.session_state.active_recorder.recording:
            audio_path = st.session_state.active_recorder.stop_recording()
            st.success("✅ Recording saved.")
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(audio_path)
                st.info("📝 Transcription: " + transcription)
            if transcription and "❌" not in transcription:
                with st.spinner("Generating response..."):
                    response = query_llm(transcription)
                    st.success(response)
                    st.session_state.chat_history.append((transcription, response, audio_path))
                    st.session_state.latest_qa = (transcription, response, audio_path)

                     # 🗣️ Speak output
                    speak(response)
                    
                st.rerun()
            else:
                st.session_state.latest_qa = (transcription, None, audio_path)
                st.error("Speech not clear or not recognized.")
        else:
            st.warning("❗ Start recording first.")


