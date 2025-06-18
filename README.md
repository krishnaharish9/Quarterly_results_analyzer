# 🧾 Quarterly Results Analyzer (RAG-Powered App)

A powerful Generative AI app that analyzes financial documents like **quarterly earnings reports**, **financial PDFs**, and **audio files from earnings calls**, enabling financial analysts to ask questions in natural language and receive accurate answers using a **RAG (Retrieval-Augmented Generation)** pipeline.

---

## 🔧 Features

- ✅ Ingests **PDFs**, **tables**, and **audio** (earnings call recordings)
- ✅ Uses **OpenAI Whisper** for audio transcription
- ✅ Combines **BM25 (keyword search)** and **dense vector search** via ChromaDB for hybrid retrieval
- ✅ Summarizes financial content and answers queries with **FLAN-T5** via LangChain
- ✅ Clean UI using **Streamlit**
- ✅ Role-based prompt engineering and few-shot examples for better accuracy

---

## 📁 Folder Structure

Quarterly_Results_Analyzer/
│
├── app.py # Streamlit frontend
├── ragpipeline.py # RAG logic and prompting
├── smart_loader.py/# PDF, table, image, and audio extractors
├── Input/ # Sample PDFs/audio for testing
├── requirements.txt # Dependencies
├── .gitignore
└── README.md


## 🧪 Example Use Cases

> After uploading a file, try asking:

- "What is the profit after tax in Q3?"
- "Summarize the key takeaways from the earnings call"
- "How did the revenue change YoY?"
- "What segments contributed most to EBITDA?"


## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/krishnaharish9/Quarterly_results_analyzer.git
cd Quarterly_results_analyzer

### 2. Creating and activating environment


python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On macOS/Linux

### 3. Installing requirements

pip install -r requirements.txt


### 4. Run the application

streamlit run app.py

### 5. How It Works (RAG Overview)
Preprocessing:

Extracts text from PDFs, images, and tables

Transcribes audio using Whisper

Chunking:

Splits text into semantically meaningful chunks

Indexing:

Stores chunks in ChromaDB with both dense (sentence-transformers) and sparse (BM25) indexes

Retrieval + Generation:

Retrieves relevant chunks using hybrid search

Combines them into a prompt for FLAN-T5 LLM to generate a response

### 6. Future Enhancements
⏳ Multi-document support

📊 Charts & metric extraction

💬 Fine-tuned financial LLM integration

☁️ Fully deployable cloud version via Render/Streamlit Cloud

### 7. 🧑‍💻 Author
Harish Krishna

• https://www.linkedin.com/in/harish-krishna-bhyravajyosula-8b35201b/
