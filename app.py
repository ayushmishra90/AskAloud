import streamlit as st
import speech_recognition as sr

# Initialize recognizer and session state
r = sr.Recognizer()
if 'listening' not in st.session_state:
    st.session_state.listening = False

def start_listening():
    st.session_state.listening = True

def stop_listening():
    st.session_state.listening = False

def recognize_speech():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.info("🎙️ Listening... Please speak.")
        try:
            audio = r.listen(source, timeout=None, phrase_time_limit=10)
            text = r.recognize_google(audio)
            st.success("📝 You said: " + text)
        except sr.UnknownValueError:
            st.error("❌ Could not understand.")
        except sr.RequestError:
            st.error("❌ API unavailable.")

# Streamlit UI
st.title("🎤 Speech to Text Converter")

col1, col2 = st.columns(2)
with col1:
    st.button("▶️ Start Speaking", on_click=start_listening)
with col2:
    st.button("⏹️ Stop Speaking", on_click=stop_listening)

if st.session_state.listening:
    recognize_speech()
