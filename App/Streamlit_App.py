import streamlit as st
import tempfile
import os
import pandas as pd
from langchain.schema import HumanMessage, AIMessage
from Hybrid_RAG import (
    ingest_pdf, process_content, chunk_text,
    embed_and_store, hybrid_retriever,
    conversational_rag, format_chat_history
)

st.set_page_config(page_icon = "🤖", page_title = "Hybrid RAG", layout = "wide")
st.title("Advanced Document Chatbot with Hybrid RAG")

def save_temp_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name
    
def initialize_rag_chain(uploaded_file):
    all_content = []
    for file in uploaded_file:
        tmp_path = save_temp_file(file)
        elements = ingest_pdf(tmp_path)
        content, images = process_content(elements)
        os.unlink(tmp_path)

        all_content.extend(content)

    chunks = chunk_text(all_content)
    vector_store = embed_and_store(chunks)
    retriever = hybrid_retriever(all_content, vector_store)
    rag_chain = conversational_rag(retriever)
    return rag_chain

uploaded_files = None

with st.sidebar:
    st.header("CONFIGURATION")
    groq_api_key = st.text_input("GROQ API Key", type = "password")
    huggingface_api_key = st.text_input("HUGGINGFACE API Key", type = "password")
    os.environ["GROQ_API_KEY"] = groq_api_key if groq_api_key else ""
    os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key if huggingface_api_key else ""
    if groq_api_key and huggingface_api_key:
        st.markdown("---")
        uploaded_files = st.file_uploader("Upload your PDF file", type = ["pdf"], accept_multiple_files=True)
    else:
        st.warning("Please enter both GROQ and HUGGINGFACE API keys to proceed.")
        
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                st.session_state.rag_chain = initialize_rag_chain(uploaded_files)
                st.session_state.chat_history = []
                st.session_state.processed_docs = len(uploaded_files)
                st.success(f"Processed {len(uploaded_files)} documents successfully!")
            except Exception as e:
                st.error(f"Failed to process documents: {str(e)}")
                if "api_key" in str(e).lower():
                    st.error("API key issue detected. Please check your API keys.")
                elif "memory" in str(e).lower():
                    st.error("Memory issue detected. Try processing smaller documents.")

    if "processed_docs" in st.session_state:
        st.markdown(f"Documents Processed: {st.session_state.processed_docs}")
        # if st.button("clear Conversation"):
        #     st.session_state.chat_history = []
        #     st.rerun()

    if st.session_state.get("rag_chain"):
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("Export Chat"):
                if st.session_state.chat_history:
                    import json
                    chat_data = {
                        "timestamp": str(st.session_state.get("processing_timestamp", "Unknown")),
                        "model": "HuggingFace Model",
                        "chunk_size": 500,
                        "chat_history": st.session_state.chat_history
                    }
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(chat_data, indent=2),
                        file_name=f"rag_chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

if st.session_state.get("rag_chain"):
    
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    if user_input := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Cooking up an answer..."):
                try:
                    formatted_history = format_chat_history(st.session_state.chat_history)
                    response = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": formatted_history
                    })
                    answer = response.get("output") or response.get("answer", "No answer generated.")
                    st.write(answer)

                    st.session_state.chat_history.append((user_input, answer))

                except Exception as e:
                    error_msg = f"Error retrieving answer: {str(e)}"
                    st.error(error_msg)

else:
    st.info("Enter the world of RAG!")

st.markdown("---")