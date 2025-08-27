import os
import traceback
import requests
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import threading
import atexit
import pyaudio
import wave
import speech_recognition as sr
import pyttsx3
import queue
import time
import platform
import shutil
import glob
from typing import Optional
import re


# LangChain / Fireworks imports
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pandas as pd
from fireworks.client import Fireworks

# ---------------------------
# Directory Management
# ---------------------------
RECORDINGS_DIR = "recordings"
DATA_DIR = "data"
VECTOR_DB_DIR = "vector_db"

# Create necessary directories
for directory in [RECORDINGS_DIR, DATA_DIR, VECTOR_DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------------------
# Audio Config for Recording
# ---------------------------
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def generate_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

# ---------------------------
# Session Cleanup Functions
# ---------------------------
def cleanup_session_files():
    """Clean up all session files"""
    try:
        # Clean recordings
        for file in glob.glob(os.path.join(RECORDINGS_DIR, "*")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing recording {file}: {e}")
        
        # Clean data folder
        for file in glob.glob(os.path.join(DATA_DIR, "*")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing data file {file}: {e}")
                
        print("Session cleanup completed")
    except Exception as e:
        print(f"Error during session cleanup: {e}")

def cleanup_on_exit():
    """Cleanup when app exits"""
    try:
        stop_speaking()
        cleanup_session_files()
        vector_path = os.path.join(VECTOR_DB_DIR, "faiss_index")
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
            print(f"Deleted vector database at: {vector_path}")
    except:
        pass

atexit.register(cleanup_on_exit)

# ---------------------------
# Speech Recording Class
# ---------------------------
class SpeechRecorder:
    def __init__(self):
        self.p = None
        self.frames = []
        self.recording = False
        self.stream = None
        self.filename = None

    def start_recording(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER
            )
            self.frames = []
            self.recording = True
            self.filename = generate_filename()
            
            print(f"Recording started: {self.filename}")
            
            while self.recording:
                try:
                    data = self.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break
                    
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.recording = False

    def stop_recording(self):
        self.recording = False
        
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
                
            if self.frames and self.filename:
                with wave.open(self.filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.frames))
                print(f"Recording saved: {self.filename}")
                return self.filename
            else:
                print("No frames to save")
                return None
                
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None

# ---------------------------
# Speech to Text
# ---------------------------
def transcribe_audio(filename):
    if not filename or not os.path.exists(filename):
        return "❌ Audio file not found."
        
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError as e:
        return f"❌ API error: {e}"
    except Exception as e:
        return f"❌ Transcription error: {e}"

# ---------------------------
# Thread-Safe Text to Speech
# ---------------------------
class ThreadSafeTTS:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()

    def _init_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)
            return True
        except Exception as e:
            print(f"TTS engine initialization failed: {e}")
            return False

    def _worker(self):
        if not self._init_engine():
            print("Failed to initialize TTS engine")
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)
                if text is None:
                    break

                if self.engine and text.strip():
                    try:
                        self.engine.stop()
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"TTS playback error: {e}")

                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}")

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        try:
            with self.lock:
                if not self.running:
                    self.running = True
                    self.worker_thread = threading.Thread(target=self._worker, daemon=True)
                    self.worker_thread.start()

            self.speech_queue.put(text)
            return True
        except Exception as e:
            print(f"Error adding to speech queue: {e}")
            return False

    def stop(self):
        with self.lock:
            if self.running:
                self.running = False
                self.speech_queue.put(None)
                if self.worker_thread and self.worker_thread.is_alive():
                    self.worker_thread.join(timeout=2)

# Global TTS instance
_tts_instance = None

def get_tts_instance():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = ThreadSafeTTS()
    return _tts_instance

def speak_system(text: str) -> bool:
    try:
        system = platform.system().lower()
        safe_text = text.replace('"', '\\"').replace("'", "\\'")
        
        if system == "windows":
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 0; $synth.Volume = 100; $synth.Speak(\'{safe_text}\')"'
            os.system(cmd)
        elif system == "darwin":
            os.system(f"say '{safe_text}'")
        elif system == "linux":
            if os.system("which espeak > /dev/null 2>&1") == 0:
                os.system(f"espeak '{safe_text}'")
            else:
                return False
        return True
    except Exception as e:
        print(f"System TTS error: {e}")
        return False

def speak_with_fallback(text: str) -> bool:
    if not text or not text.strip():
        return False
        
    try:
        tts = get_tts_instance()
        if tts.speak(text):
            return True
    except Exception as e:
        print(f"Primary TTS error: {e}")
    
    return speak_system(text)

def stop_speaking():
    global _tts_instance
    try:
        if _tts_instance:
            _tts_instance.stop()
            _tts_instance = None
    except Exception as e:
        print(f"Error stopping TTS: {e}")

# ---------------------------
# Enhanced Document Processing for Tables
# ---------------------------
def process_document_with_tables(file_path):
    """Enhanced document processing that better handles tables"""
    documents = []
    
    try:
        # Load document with Docling
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        
        # Process each document
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata
            
            # Enhanced table detection and formatting
            if 'table' in content.lower() or '|' in content or '\t' in content:
                # This looks like it contains tabular data
                lines = content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # Clean and format table-like content
                    if '|' in line or '\t' in line:
                        # Replace multiple spaces/tabs with single spaces
                        cleaned_line = ' '.join(line.split())
                        processed_lines.append(f"TABLE ROW: {cleaned_line}")
                    else:
                        processed_lines.append(line)
                
                content = '\n'.join(processed_lines)
            
            # Add document with enhanced metadata
            enhanced_doc = Document(
                page_content=content,
                metadata={
                    **metadata,
                    'contains_tables': '|' in content or 'TABLE ROW:' in content,
                    'file_path': file_path,
                    'processed_time': datetime.now().isoformat()
                }
            )
            documents.append(enhanced_doc)
            
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []
    
    return documents

# ---------------------------
# Vector Database Management
# ---------------------------

def initialize_or_load_vectorstore(embeddings):
    """Initialize new vectorstore or load existing one, ensuring dimension match."""
    vector_path = os.path.join(VECTOR_DB_DIR, "faiss_index")

    if os.path.exists(vector_path) and os.listdir(vector_path):
        try:
            vectorstore = FAISS.load_local(
                vector_path, embeddings, allow_dangerous_deserialization=True
            )

            # ✅ Check dimension match
            sample_vec = embeddings.embed_query("dimension test")
            if len(sample_vec) != vectorstore.index.d:
                print(
                    f"Dimension mismatch! "
                    f"Index expects {vectorstore.index.d}, embeddings give {len(sample_vec)}. "
                    f"Rebuilding vectorstore..."
                )
                shutil.rmtree(vector_path)  # delete old index
                return None, False

            return vectorstore, True
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}")
            return None, False
    else:
        return None, False


def add_documents_to_vectorstore(documents, embeddings, existing_vectorstore=None):
    """Add new documents to existing vectorstore or create new one."""
    vector_path = os.path.join(VECTOR_DB_DIR, "faiss_index")

    if existing_vectorstore is not None:
        try:
            existing_vectorstore.add_documents(documents)
            existing_vectorstore.save_local(vector_path)
            return existing_vectorstore
        except Exception as e:
            print(f"Error adding documents: {e}")
            # Fallback: rebuild new index
            shutil.rmtree(vector_path)
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(vector_path)
            return vectorstore
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(vector_path)
        return vectorstore


# ---------------------------
# Context relevance checker
# ---------------------------
def is_context_relevant(query: str, context: str, min_similarity=0.2):
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not context.strip():
        return False
    
    # Check for table-related queries
    table_keywords = {'table', 'row', 'column', 'data', 'value', 'list', 'compare', 'vs', 'versus'}
    if any(keyword in query.lower() for keyword in table_keywords):
        return True
    
    overlap = len(query_words.intersection(context_words))
    similarity = overlap / len(query_words) if query_words else 0
    
    return similarity >= min_similarity or len(context.strip()) > 30

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AskAloud", page_icon="🎤")
st.title("🎙️ AskAloud: AI Voice Assistant with Enhanced Table Reading")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = True

# Load environment variables
load_dotenv()
FA_TOKEN = os.getenv("FA_TOKEN")

# Initialize LLM and embeddings if not done
if st.session_state.llm is None and FA_TOKEN:
    st.session_state.llm = Fireworks(api_key=FA_TOKEN)
    st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cuda"}
                )
    
    # Try to load existing vectorstore
    vectorstore, loaded = initialize_or_load_vectorstore(st.session_state.embeddings)
    if loaded:
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        st.success("📚 Previous documents loaded from vector database!")

# ---------------------------
# Speech Control
# ---------------------------
st.sidebar.header("🔊 Speech Settings")
speech_toggle = st.sidebar.toggle(
    "Enable Speech Response", 
    value=st.session_state.speech_enabled,
    help="Toggle AI voice responses on/off"
)
st.session_state.speech_enabled = speech_toggle

if st.session_state.speech_enabled:
    st.sidebar.success("🔊 Speech ON")
else:
    st.sidebar.info("🔇 Speech OFF")

# ---------------------------
# Session Management
# ---------------------------
st.sidebar.header("🗂️ Session Management")
if st.sidebar.button("🗑️ Clear All Data", help="Clear all documents and start fresh"):
    cleanup_session_files()
    
    # Reset session state
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    st.session_state.documents_loaded = []
    st.session_state.current_answer = None
    
    # Remove vector database
    vector_path = os.path.join(VECTOR_DB_DIR, "faiss_index")
    if os.path.exists(vector_path):
        shutil.rmtree(vector_path)
    
    st.sidebar.success("🗑️ All data cleared!")
    st.rerun()

# Display loaded documents
if st.session_state.documents_loaded:
    st.sidebar.subheader("📑 Loaded Documents")
    for i, doc_name in enumerate(st.session_state.documents_loaded, 1):
        st.sidebar.text(f"{i}. {doc_name}")

# ---------------------------
# File & Link Loader
# ---------------------------
st.header("📂 Add Documents to Knowledge Base")

# User chooses mode
mode = st.radio("Choose input type:", ["Upload File", "Enter Link"])

def process_and_add_document(file_path, doc_name):
    """Process and add document to vectorstore"""
    if not FA_TOKEN:
        st.error("❌ FA_TOKEN not found in environment variables.")
        return False
        
    try:
        with st.spinner(f"Processing {doc_name}..."):
            # Process document
            documents = process_document_with_tables(file_path)
            
            if not documents:
                st.error("❌ No content extracted from document")
                return False

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            # chunks = splitter.split_documents(documents)
            chunks = []
            for doc in documents:
                split_texts = splitter.split_text(doc.page_content)
                for i, text in enumerate(split_texts):
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                **doc.metadata,
                                "chunk_id": str(uuid.uuid4()),
                                "chunk_index": i,
                                "source_file": doc_name
                            }
                        )
                    )

            # Add to vectorstore
            st.session_state.vectorstore = add_documents_to_vectorstore(
                chunks, 
                st.session_state.embeddings, 
                st.session_state.vectorstore
            )
            
            # Update retriever
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

            # Track loaded documents
            if doc_name not in st.session_state.documents_loaded:
                st.session_state.documents_loaded.append(doc_name)
            
            st.success(f"✅ {doc_name} added to knowledge base!")
            return True
            
    except Exception as e:
        st.error(f"❌ Error processing {doc_name}: {e}")
        # st.code(traceback.format_exc())
        return False


if mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, PPT, HTML)",
        type=["pdf", "docx", "ppt", "pptx", "html"]
    )
    if uploaded_file:
        # Save to data folder
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if st.button(f"Add '{uploaded_file.name}' to Knowledge Base"):
            process_and_add_document(file_path, uploaded_file.name)

elif mode == "Enter Link":
    link = st.text_input("Enter document link (http/https)")
    if link:
        if st.button("Add Link to Knowledge Base"):
            try:
                # filename = link.split("/")[-1] or "document.html"
                safe_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', link)

                # Try to keep extension from URL if any
                if "." in os.path.basename(link):
                    filename = safe_name
                else:
                    filename = f"{safe_name}.html"
                file_path = os.path.join(DATA_DIR, filename)
                
                with st.spinner("Downloading..."):
                    response = requests.get(link, timeout=10)
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                
                process_and_add_document(file_path, filename)
                
            except Exception as e:
                st.error(f"❌ Failed to download from link: {e}")

# ---------------------------
# Enhanced query function
# ---------------------------
def ask_document(query: str):
    retriever = st.session_state.retriever
    llm = st.session_state.llm

    if not retriever or not llm:
        return "❌ Please add at least one document to the knowledge base first."

    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Check if context is relevant
        if not is_context_relevant(query, context):
            return "❌ I couldn't find relevant information in the loaded documents to answer your question. Please ask questions related to the document content."
        
        # Enhanced prompt for better table handling
        prompt = f"""
You are a helpful AI assistant that answers questions conversationally 
using the context retrieved from the documents or website. 

### Rules:
1. Always ground your answers in the provided context, but interpret meaningfully — 
   don’t just copy text.
2. If multiple chunks are relevant, combine them into a single clear answer.
3. Be concise and conversational, not robotic.
4. For lists, tables, or comparisons, format the output clearly in Markdown.
5. If the context doesn’t fully answer the question, say what is missing and 
   suggest a possible next step (e.g., “This detail is not in the documents, 
   you may need to check XYZ section”).
6. Use prior chat history to maintain continuity in conversation.
7. Never hallucinate or invent facts not supported by the documents.


CONTEXT:
{context}

QUESTION: {query}

ANSWER (based only on the provided context, with special attention to tabular data):"""

        # Use Fireworks API for completion
        response = llm.chat.completions.create(
            model="accounts/fireworks/models/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        # if not context.strip():
        if is_context_relevant(query, context, min_similarity=0.3) is False:
            return "❌ No relevant context found in the documents for your question."
            
        return answer

    except Exception as e:
        return f"❌ Error processing your query: {str(e)}"


# ---------------------------
# Conversation History
# ---------------------------
st.subheader("📝 Conversation History")
if st.session_state.chat_history:
    for i, (q, r, path) in enumerate(st.session_state.chat_history, 1):
    # for i, (q, r, path, metas) in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Q{i}: {q[:50]}...",expanded=True):
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Answer:** {r}")


            # Show audio if present
            if path and os.path.exists(path):
                st.audio(path, format="audio/wav")

            # # Show which chunks were used
            # st.write("**📑 Context Chunks Used:**")
            # for m in metas:
            #     st.markdown(f"""
            #     - **Source File:** `{m.get("source_file", "Unknown")}`
            #     - **Chunk ID:** `{m.get("chunk_id", "N/A")}`
            #     - **Chunk Index:** `{m.get("chunk_index", "N/A")}`
            #     """)
        st.markdown("---")
else:
    st.info("No conversations yet. Add documents and start asking questions!")

# ---------------------------
# Statistics and Tips
# ---------------------------
col1, = st.columns(1)

with col1:
    if st.session_state.retriever:
        st.metric("📚 Documents Loaded", len(st.session_state.documents_loaded))
        st.metric("💬 Questions Asked", len(st.session_state.chat_history))
    # with st.expander("💡 Tips for Better Results"):
    #     st.markdown("""
    #     **For better results with tables:**
    #     - "What is the value of X in the table?"
    #     - "Compare A and B from the data"
    #     - "List all items where condition Y is met"
    #     - "What are the highest/lowest values?"
        
    #     **Document Management:**
    #     - Add multiple documents to build comprehensive knowledge
    #     - Use specific terminology from your documents
    #     - Ask follow-up questions for clarification
    #     """)


# ---------------------------
# Manual text input option
# ---------------------------
st.subheader("✏️ Ask Your Question")
text_query = st.text_input("Ask about tables, data, or any document content:")
if text_query and st.button("Ask Question"):
    if st.session_state.retriever:
        with st.spinner("🤖 Analyzing documents and generating answer..."):
            response = ask_document(text_query)
            
            # Store current answer
            st.session_state.current_answer = response
            
            # Add to chat history
            st.session_state.chat_history.append((text_query, response, None))
            
            # Speech response if enabled
            if st.session_state.speech_enabled and "❌" not in response:
                with st.spinner("🔊 Speaking response..."):
                    try:
                        speak_with_fallback(response)
                    except Exception as e:
                        st.warning(f"⚠️ Speech error: {e}")
        
        st.rerun()
    else:
        st.warning("Please add at least one document to the knowledge base first!")

# ---------------------------
# Record & Ask Section
# ---------------------------
st.subheader("🎤 Voice Input")
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Recording"):
        try:
            if hasattr(st.session_state, 'active_recorder'):
                del st.session_state.active_recorder
            
            stop_speaking()
            st.session_state.active_recorder = SpeechRecorder()
            st.session_state.thread = threading.Thread(
                target=st.session_state.active_recorder.start_recording
            )
            st.session_state.thread.start()
            st.info("🎙️ Recording... Press Stop when done.")
        except Exception as e:
            st.error(f"❌ Failed to start recording: {e}")

with col2:
    if st.button("⏹️ Stop & Ask"):
        if st.session_state.get("active_recorder") and st.session_state.active_recorder.recording:
            try:
                audio_path = st.session_state.active_recorder.stop_recording()
                
                if audio_path:
                    st.success("✅ Recording saved.")

                    with st.spinner("🔎 Transcribing..."):
                        transcription = transcribe_audio(audio_path)
                        st.info(f"📝 Transcription: {transcription}")

                    if transcription and "❌" not in transcription:
                        if st.session_state.retriever:
                            with st.spinner("🤖 Analyzing documents and generating answer..."):
                                response = ask_document(transcription)
                                
                                # Store current answer
                                st.session_state.current_answer = response
                                
                                # Add to chat history
                                st.session_state.chat_history.append((transcription, response, audio_path))

                                # Speech response if enabled
                                if st.session_state.speech_enabled and "❌" not in response:
                                    with st.spinner("🔊 Speaking response..."):
                                        try:
                                            speak_with_fallback(response)
                                        except Exception as speech_error:
                                            st.warning(f"⚠️ Speech error: {speech_error}")
                            st.rerun()
                        else:
                            st.warning("❗ Please add at least one document to the knowledge base first!")
                    else:
                        st.error("❌ Speech not clear or not recognized. Please try again.")
                else:
                    st.error("❌ Failed to save recording.")
            except Exception as e:
                st.error(f"❌ Error processing recording: {e}")
        else:
            st.warning("❗ Start recording first.")