import streamlit as st
from smart_loader import smart_ingest_files
from ragpipeline import setup_rag_pipeline_from_docs

st.title("📊 Quarterly Report Q&A Assistant")

# Upload and ingest files
uploaded_files = st.file_uploader("Upload PDFs or audio files", accept_multiple_files=True, type=["pdf", "mp3", "wav", "m4a"])

if uploaded_files:
    file_info = []

    # Save files locally and prepare (filepath, label) pairs
    for uploaded_file in uploaded_files:
        filepath = f"temp_uploads/{uploaded_file.name}"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())
        file_info.append((filepath, uploaded_file.name))

    with st.spinner("🔍 Ingesting and indexing files..."):
        documents = smart_ingest_files(file_info)
        qa_chain = setup_rag_pipeline_from_docs(documents)

    st.success("✅ Files processed. Ask your questions below!")

    # Ask questions
    user_question = st.text_input("Ask a question about the reports:")

    if user_question:
        with st.spinner("💬 Generating answer..."):
            answer = qa_chain.invoke({"query": user_question})
            st.markdown(f"**Answer:** {answer['result']}")