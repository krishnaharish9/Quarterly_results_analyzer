import os
import gc
import time
import camelot
import pdfplumber
import whisper
from langchain.schema import Document

def extract_tables(path: str) -> list[str]:
    """
    Extracts tables from a PDF using Camelot (stream flavor to avoid Ghostscript).
    Adds delay and garbage collection to prevent PermissionError on Windows.
    """
    try:
        tables = camelot.read_pdf(path, pages='all', flavor='stream')  # Avoid Ghostscript
        extracted = [t.df.to_string(index=False) for t in tables if not t.df.empty]
        
        # Help Windows release file locks
        gc.collect()
        time.sleep(0.5)
        
        return extracted
    except Exception as e:
        print(f"[Table Extraction Failed] {e}")
        return []

def extract_pdf(path: str, label: str) -> list[Document]:
    """
    Extracts text and tables from a machine-readable PDF.
    """
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": label,
                            "type": "text",
                            "page": i + 1,
                            "filename": os.path.basename(path)
                        }
                    ))
    except Exception as e:
        print(f"[PDF Text Extraction Failed] {e}")

    table_texts = extract_tables(path)
    for tbl in table_texts:
        docs.append(Document(
            page_content=tbl,
            metadata={
                "source": label,
                "type": "table",
                "filename": os.path.basename(path)
            }
        ))

    return docs

def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Transcribes audio file using OpenAI Whisper.
    """
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"[Audio Transcription Failed] {e}")
        return ""

def smart_ingest_files(file_info_list: list[tuple[str, str]]) -> list[Document]:
    """
    Accepts a list of (file_path, label) tuples.
    Returns a list of LangChain Documents.
    """
    all_docs = []

    for file_path, label in file_info_list:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in [".mp3", ".wav", ".m4a"]:
            print(f"Transcribing audio: {file_path}")
            text = transcribe_audio(file_path)
            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": label,
                        "type": "audio",
                        "filename": os.path.basename(file_path)
                    }
                ))

        elif ext == ".pdf":
            print(f"Processing PDF: {file_path}")
            docs = extract_pdf(file_path, label)
            all_docs.extend(docs)

        else:
            print(f"[Unsupported File Type] Skipped: {file_path}")

    return all_docs


# if __name__ == "__main__":
#     files = [
#         ("C:/Users/krish/Desktop/Quarterly_Results_Analyzer/Input/ICICI/analyst-call-transcript-q4-2025.pdf", "conference_call"),
#         ("C:/Users/krish/Desktop/Quarterly_Results_Analyzer/Input/ICICI/icici-bank-financial-results-q4-2025.pdf", "financial_report")
#     ]

#     docs = smart_ingest_files(files)
    
#     for doc in docs:
#         # print(doc.metadata)

#         print(doc.page_content, "\n---")