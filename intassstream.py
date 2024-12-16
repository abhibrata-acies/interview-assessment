import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

image_path = "aciesglobal.png"  # Path to the static image
image = Image.open(image_path)
st.image(image, caption="Your Static Image", use_column_width=True)

# Step 1: Set up your APIs and environment variables
os.environ['GROQ_API_KEY'] = ""

# Step 2: Initialize the language model and embeddings
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Define functions for processing PDF and generating assessments
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def get_interview_prompt(docs):
    return f"""
    You are an interviewer. Based on the interview conversation provided below, generate an assessment of the candidate based on aspects only- 
    1. "First principle thinking",
    2. "Industry experience",
    3. "Statistics knowledge",
    4. "Technical expertise",
    5. "Communication",
    6. "Leadership" 
    
    Resume:
    {docs}

    Please provide the interview questions below:
    """

def generate_interview_assessment(llm, docs):
    prompt = get_interview_prompt(docs)
    response = llm.predict(prompt)
    return response

# Step 4: Deploy in Streamlit
st.title("Candidate Assessment Tool")

uploaded_file = st.file_uploader("Upload the interview transcript (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing the PDF..."):
        # Save the uploaded file temporarily
        temp_file_path = "uploaded_transcript.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the PDF
        splits = load_and_split_pdf(temp_file_path)

        # Convert split documents back into a single text for prompt
        concatenated_text = "\n".join([doc.page_content for doc in splits])

        # Generate assessment
        assessment = generate_interview_assessment(llm, concatenated_text)

        # Display the results
        st.header("Assessment Results")
        st.text_area("Assessment:", assessment, height=300)
