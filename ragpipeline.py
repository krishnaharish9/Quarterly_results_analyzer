import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

CHROMA_DB_DIR = "chroma_db"

# Load local LLM model via transformers
def load_local_llm():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

# Prompt
fewshot_prompt = """
You are an AI assistant specializing in analyzing quarterly financial results. 
Your task is to extract key insights, answer questions concisely, and present data-driven conclusions.

Example 1:
Input: What was the revenue growth in Q3?
Answer: The revenue grew by 15% in Q3, reaching $3.2 billion compared to $2.78 billion in Q2.

Example 2:
Input: How did the operating margin change?
Answer: Operating margin improved to 18.5% from 17.2% due to cost optimization.

Now respond to the following input based on context from quarterly reports:
Input: {input}
Answer:
"""

prompt = PromptTemplate(input_variables=["input"], template=fewshot_prompt.strip())

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def get_hybrid_retriever(vectordb, docs):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    dense = vectordb.as_retriever(search_kwargs={"k": 3})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )
    return hybrid

def create_qa_chain(llm, retriever):
    stuff_chain = StuffDocumentsChain(llm_chain=LLMChain(llm=llm, prompt=prompt), document_variable_name="input")
    return RetrievalQA(retriever=retriever, combine_documents_chain=stuff_chain)

def setup_rag_pipeline_from_docs(docs):
    chunks = chunk_documents(docs)
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    filtered = filter_complex_metadata(chunks)

    vectordb = Chroma.from_documents(filtered, embedding=embedding, persist_directory=None)
    retriever = get_hybrid_retriever(vectordb, chunks)
    llm = load_local_llm()

    return create_qa_chain(llm, retriever)