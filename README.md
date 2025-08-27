# 🎙️ AskAloud — Intelligent Document Q&A Chatbot  

## 📌 Overview  
**AskAloud** is an AI-powered chatbot that allows you to **ask questions about your documents** and get **accurate, conversational answers**.  
It uses **LLMs + embeddings + vector search** to understand your queries in context, and even responds with **speech output**, making the experience more natural.  

This project demonstrates how to build a **retrieval-augmented generation (RAG) system** with:  
- Context-aware document processing  
- Vector database storage  
- Interactive voice-based Q&A  

---

## 🚀 Features  
- ✅ **Context-aware Q&A** — Upload PDFs, DOCX, PPTX, HTML files and ask questions based on them  
- ✅ **LLM-powered responses** — Uses Fireworks AI + Hugging Face embeddings for precise answers  
- ✅ **Vector search with FAISS** — Efficient semantic retrieval of relevant chunks  
- ✅ **Voice interaction** — Ask via microphone, get spoken responses  
- ✅ **Conversation history** — Maintains context across questions  
- ✅ **Simple & interactive UI** — Built with Streamlit  

---

## 🛠️ Tech Stack  

### Core
- **Python 3.10+**
- **Streamlit** → Web-based interactive interface  
- **LangChain** → Document parsing, splitting, RAG pipeline  
- **FAISS** → Vector database for semantic search  
- **Hugging Face Embeddings** → High-quality text embeddings  
- **Fireworks AI API** → LLM backend for Q&A  

### Speech
- **SpeechRecognition** → Convert voice → text  
- **pyttsx3** → Convert text → speech  

### Document Processing 
- **Docling** → Enhanced parsing for all provided data files type  

---

## 📂 Project Structure 
```st
askaloud/
├── .vscode/                  # VSCode workspace settings
├── .gitignore                # Git ignore rules
├── all_installed.txt         # List of installed packages (for reference)
├── basic_final.py            # Main app script (Streamlit chatbot)
├── basic.py                  # Auxiliary script (starting point)
├── basic2.py                 # Auxiliary script
├── input.py                  # Basic input for LLM
├── inputf.py                 # Final input for LLM (Fireworks)
├── installed.txt             # Installed packages reference
├── llm.py                    # LLM setup
├── record.py                 # Audio recording utilities
├── sp_to_txt.py              # Speech-to-text utilities
└── speak.py                  # Text-to-speech utilities

# Generated after usage
├── data/                     # Uploaded documents and downloaded files
├── env/                      # Python virtual environment
├── recordings/               # Saved audio recordings
├── vector_db/                # FAISS vector database for embeddings
└── .env                      # Environment variables (FA_TOKEN etc.)
```

---

## ⚡ Installation  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/ayushmishra90/AskAloud.git
cd AskAloud
```
### 2️⃣ Create virtual environment & install dependencies
```bash
python -m venv env
source env/bin/activate   # On Linux/Mac
env\Scripts\activate      # On Windows

pip install -r installed.txt
```
### 3️⃣ Setup environment variables
- Get your token from
  - Go to Fireworks AI and log in.
  - Go to Account → API Keys.
  - Click Create API Key and name it.
  - Copy the key immediately and store it safely.
  - Use it in your code or .env file:
- Create a .env file in the root:
```.env
FA_TOKEN=your_fireworks_api_key
```
### 4️⃣ Run the app
```bash
streamlit run basic_final.py
```

---
## 🎯 Usage

### 🔹 Upload & Ask
- Upload a **PDF/DOCX/PPTX/HTML** file or **links** .
- The document is parsed, split into chunks, embedded, and stored in **FAISS vector DB**.
- Ask questions about the document → The chatbot fetches relevant chunks and generates a contextual answer.

### 🔹 Voice Interaction
- Click **🎤 Speak** → Ask a question out loud.
- The bot will transcribe → search → respond **both in text & speech**.

### 🔹 Clear Knowledge Base
- Use **🗑️ Clear Data** → Removes all stored embeddings & resets the system.

---

## 📊 Example Workflow

1. **Upload Resume**  
   Upload `resume.pdf` containing basic details about education, skills, and extracurricular activities.

2. **Ask Question**  
   Example: *"What extracurricular activities does the candidate have?"*  
   - Bot retrieves relevant sections from the resume → Summarizes → Provides an answer.

3. **Upload Updated Resume**  
   Upload `resume_updated.pdf` which now includes more extracurricular activities.

4. **Ask Follow-up Question**  
   Example: *"What extracurricular activities does the candidate have?"*  
   - Bot now retrieves from the updated knowledge base → Combines previous and new information → Provides a **more detailed answer** reflecting all activities.

5. **Add Website Content**  
   Enter a website URL (e.g., `https://example.com/article`) in the sidebar.  
   - Bot downloads the web page → Parses text content → Embeds it into the knowledge base.

6. **Ask Questions on Website Content**  
   Example: *"What are the key points of this article?"*  
   - Bot fetches relevant content from the web page → Summarizes → Answers contextually.

7. **Ask Topic-Specific Question on Website**  
   Example: *"Explain the algorithm discussed on this page."*  
   - Bot searches the web page content → Extracts algorithm details → Provides a clear explanation.

8. **Compare Across Sources**  
   Example: *"Compare this article’s algorithm with previous documents."*  
   - Bot searches all uploaded documents and web content → Combines relevant info → Provides a comprehensive, context-aware answer.


---

## 🔧 Future Improvements
- 📈 Visualization of retrieved chunks for better insight into what the AI is referencing.
- 🌍 Deploy on **Streamlit Cloud / HuggingFace Spaces** for easy access and sharing.
- 🎙️ More natural **TTS voice integration** for realistic speech output.
- 🎤 **Enhanced Voice Interaction & Interpretation**  
  - Allow the bot to handle conversational follow-ups via voice.
  - Support multiple languages and accents.
  - Improve transcription accuracy and speed for real-time Q&A.

---

## Demo Images
<img width="1917" height="965" alt="Screenshot 2025-08-27 231621" src="https://github.com/user-attachments/assets/b7960eb2-e49c-41dd-b6e6-a527b507a40a" />
<img width="1880" height="875" alt="Screenshot 2025-08-27 231713" src="https://github.com/user-attachments/assets/7c9bb2d2-fe30-4fbf-8151-5c08d4d9fab2" />
<img width="1918" height="969" alt="Screenshot 2025-08-27 231731" src="https://github.com/user-attachments/assets/fc0ce5ad-aaed-4fdb-8900-66201ef7ce16" />
<img width="1885" height="884" alt="Screenshot 2025-08-27 231745" src="https://github.com/user-attachments/assets/2b2e71e6-caa1-4c0f-8ead-cfe4ce2dd4e0" />
<img width="1881" height="880" alt="Screenshot 2025-08-27 231820" src="https://github.com/user-attachments/assets/78ae0057-34ec-4f8d-9f43-eeeb0d79dd43" />
<img width="1880" height="881" alt="Screenshot 2025-08-27 231831" src="https://github.com/user-attachments/assets/503c4cb9-7f57-4f43-a348-e2af7001c162" />
<img width="1870" height="892" alt="Screenshot 2025-08-27 231852" src="https://github.com/user-attachments/assets/4fadbde4-9bca-4ecb-b03a-710e77dc70b9" />
<img width="1882" height="875" alt="Screenshot 2025-08-27 231915" src="https://github.com/user-attachments/assets/4aebfa0d-0c8c-4f57-b1c5-af656c6fc233" />


## 📜 License
MIT License © 2025

