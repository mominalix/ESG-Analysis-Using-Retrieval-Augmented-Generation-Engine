import os
import tempfile
import pinecone
import openai
from pathlib import Path
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import spacy
import json
import pypdf2

# Configure API keys and environment
openai.api_key = "YOUR_OPENAI_API_KEY"
pinecone.init("YOUR_PINECONE_API_KEY", environment="YOUR_ENVIRONMENT")

# Configuration
index_name = "esg-rag-index"  # Index name for Pinecone
source_type = "json"  # Source type for regulations (e.g., JSON, database)

# Helper functions
def load_policy_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        reader = pypdf2.PdfReader(file)
        policy_text = ""
        for page in reader.pages:
            policy_text += page.extract_text()
    return policy_text

def clean_policy_text(text):
    # Placeholder for text cleaning logic
    return text

def load_regulations_from_source():
    if source_type == "json":
        with open("regulations.json") as file:
            return json.load(file)
    # Add loaders for other source types if needed

def extract_policy(source_documents):
    # Placeholder for policy extraction logic
    return source_documents[0].metadata['policy_text']

def generate_analysis(policy_text, regulation_text):
    prompt = f"Analyze the following policy against the regulation. Identify gaps or discrepancies. Policy: {policy_text} Regulation: {regulation_text}"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=400)
    return response.choices[0].text

def process_response(response):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(response)
    gaps = []
    for entity in doc.ents:
        if entity.label_ in ["REGULATION", "POLICY_ELEMENT"]:
             gaps.append((entity.text, entity.label_, "potential_gap"))
    return gaps

# Main Analysis Logic
document_store = Chroma.from_documents(regulations, OpenAIEmbeddings())  # Use Chroma vector store for documents
retriever = document_store.as_retriever()

# Streamlit UI
def input_fields():
    # Streamlit sidebar for input fields
    with st.sidebar:
        st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        st.session_state.pinecone_env = st.text_input("Pinecone environment")
        st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def process_documents():
    # Process uploaded documents
    try:
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
                tmp_file_path = tmp_file.name
            policy_text = load_policy_from_pdf(tmp_file_path)
            cleaned_policy_text = clean_policy_text(policy_text)
            st.write("Policy Text:", cleaned_policy_text)
            # Perform analysis
            results = retriever.retrieve(cleaned_policy_text, top_k=3)
            for result in results:
                regulation_text = result.text
                st.write("Regulation Text:", regulation_text)
                analysis_result = generate_analysis(cleaned_policy_text, regulation_text)
                gaps = process_response(analysis_result)
                st.write("Potential Gaps:")
                st.write(gaps)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config(page_title="ESG RAG")
    st.title("ESG Retrieval Augmented Generation (RAG) Tool")
    input_fields()
    st.button("Submit Documents", on_click=process_documents)

if __name__ == '__main__':
    main()
