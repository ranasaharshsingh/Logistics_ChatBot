import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# New way
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load the API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Function to Read PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 3. Function to Chunk Text (Break it into small pieces)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# ... (Functions 1, 2, 3 remain the same)

# 4. Function to Create Vector DB (The Memory)
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # FAISS.from_texts creates a fresh database every time. 
    # If you want to ADD to an existing one, you would use merge_from, 
    # but for a fresh upload session, collecting all chunks here is correct.
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# 5. Function to Process User Questions
def process_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if not os.path.exists("faiss_index"):
        return "⚠️ No documents found. Please upload and 'Process' your PDFs first!"

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # CRITICAL CHANGE: Increase 'k' so it finds chunks from DIFFERENT documents.
        # If you have multiple docs, k=10 or k=15 is much safer than the default.
        docs = new_db.similarity_search(user_question, k=10) 
        
    except Exception as e:
        return f"System Error: Could not read the index. Error: {e}"

    prompt_template = """
    You are an expert Logistics Manager. Answer the question based ONLY on the provided context.
    The context below contains information from multiple logistics documents.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]