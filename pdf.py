import os
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API key from the .env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please set it in the .env file.")
else:
    genai.configure(api_key=api_key)


# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_docs):
    """Extracts all text from the given PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text


# Step 2: Split extracted text into chunks for efficient processing
def split_text_into_chunks(text):
    """Splits text into manageable chunks for vector embeddings."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_text(text)
    return chunks


# Step 3: Embed the text chunks using Google Generative AI and FAISS
def create_vector_store(text_chunks):
    """Embeds text chunks using GoogleGenerativeAIEmbeddings and creates FAISS vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")


# Step 4: Generate the prompt and QA chain using Google Gemini Pro
def get_qa_chain():
    """Creates a Question Answering chain using Google Gemini Pro."""
    prompt_template = """
    Use the following context to answer the question comprehensively. 
    Focus on information related to future growth, business changes, triggers, and next year's earnings and growth.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Step 5: Process user input and generate response
def process_user_question(user_question):
    """Handles the user question by searching the FAISS vector store and generating answers."""
    try:
        # Initialize the Google Generative AI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load FAISS vector store and enable dangerous deserialization
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search based on user question
        docs = vector_store.similarity_search(user_question)

        # Get the QA chain using Google Gemini Pro
        chain = get_qa_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Answer: ", response.get("output_text", "No answer found."))
    except Exception as e:
        st.error(f"Error during question answering: {e}")



# Streamlit app main logic
def main():
    st.set_page_config(page_title="Investor Document Analysis", page_icon="📊")
    st.title("Company Document Analysis for Investors 📈")

    st.write("This app helps investors evaluate company documents by extracting key information such as future growth, business changes, and material events.")

    # Upload PDF files
    uploaded_pdfs = st.sidebar.file_uploader("Upload your PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_pdfs and st.sidebar.button("Process PDFs"):
        with st.spinner("Extracting and processing PDF content..."):
            raw_text = extract_text_from_pdf(uploaded_pdfs)
            if raw_text:
                text_chunks = split_text_into_chunks(raw_text)
                create_vector_store(text_chunks)
                st.sidebar.success("Processing completed!")

    # Get user question input
    user_question = st.text_input("Ask a question based on the PDF content")
    if user_question:
        process_user_question(user_question)


if __name__ == "__main__":
    main()
