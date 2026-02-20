import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load the API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
#  IMPROVED VALIDATION - Better accuracy, minimal API usage
# ============================================================

# High-priority keywords (strong indicators of logistics docs)
HIGH_PRIORITY_KEYWORDS = [
    'bill of lading', 'b/l', 'bol', 'lorry receipt', 'airway bill', 'awb',
    'sea freight', 'air freight', 'shipping document', 'freight forwarder',
    'container number', 'seal number', 'booking number', 'bl number',
    'shipment', 'consignee', 'consignor', 'shipper', 'carrier'
]

# Medium-priority keywords
MEDIUM_PRIORITY_KEYWORDS = [
    'invoice', 'commercial invoice', 'tax invoice', 'packing list',
    'delivery note', 'dispatch note', 'purchase order', 'po number',
    'warehouse', 'warehousing', 'inventory', 'stock', 'logistics',
    'supply chain', 'procurement', 'freight', 'cargo', 'carrier',
    'transport', 'transportation', 'truck', 'vehicle', 'driver',
    'customs', 'clearance', 'duty', 'hs code', 'import', 'export'
]

# Low-priority keywords
LOW_PRIORITY_KEYWORDS = [
    'eta', 'etd', 'arrival', 'departure', 'transit', 'route',
    'fob', 'cif', 'cfr', 'incoterms', 'port of loading', 'port of discharge',
    'delivery', 'dispatch', 'manifest', 'certificate', 'license',
    'sales order', 'so number', 'goods', 'merchandise', 'booking'
]


def calculate_keyword_score(text):
    """Calculate weighted keyword score"""
    text_lower = text.lower()
    score = 0
    
    # High priority = 3 points each
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in text_lower:
            score += 3
    
    # Medium priority = 2 points each
    for kw in MEDIUM_PRIORITY_KEYWORDS:
        if kw in text_lower:
            score += 2
    
    # Low priority = 1 point each
    for kw in LOW_PRIORITY_KEYWORDS:
        if kw in text_lower:
            score += 1
    
    return score


def get_found_keywords(text):
    """Get list of found keywords"""
    text_lower = text.lower()
    found = []
    
    for kw in HIGH_PRIORITY_KEYWORDS + MEDIUM_PRIORITY_KEYWORDS + LOW_PRIORITY_KEYWORDS:
        if kw in text_lower and kw not in found:
            found.append(kw)
    
    return found[:6]  # Return top 6


def validate_logistics_document_ai(text, model_name="gemini-2.0-flash"):
    """
    IMPROVED: Uses keyword validation FIRST, only uses AI for borderline cases
    """
    
    score = calculate_keyword_score(text)
    found = get_found_keywords(text)
    
    # Clear cases - no AI needed
    if score >= 5:
        # Definitely valid - high confidence
        return True, f"✅ Valid logistics document. Found: {', '.join(found)}"
    
    if score == 0:
        # Definitely not valid - no logistics keywords
        return False, "⚠️ This document doesn't appear to be a logistics document."
    
    # Borderline cases (score 1-4) - use AI to confirm
    try:
        sample_text = text[:1200]  # Smaller sample
        
        validation_prompt = f"""
        Is this a logistics document? Consider: invoices, bills of lading, 
        shipping docs, packing lists, purchase orders, delivery notes, freight docs.
        
        Reply with only YES or NO.
        
        Document preview: {sample_text[:800]}
        """
        
        model = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.1,
            google_api_key=API_KEY
        )
        response = model.invoke(validation_prompt)
        response_text = response.content.strip().upper()
        
        if "YES" in response_text:
            return True, f"✅ Valid logistics document (verified). Found: {', '.join(found)}"
        else:
            return False, "⚠️ This document doesn't appear to be a logistics document."
            
    except Exception as e:
        print(f"AI Validation Error: {e}")
        # Fallback to keyword result
        if score >= 3:
            return True, f"✅ Valid logistics document. Found: {', '.join(found)}"
        else:
            return False, "⚠️ This document doesn't appear to be a logistics document."


def is_logistics_document(text):
    """Simple keyword validation (for fallback)"""
    score = calculate_keyword_score(text)
    found = get_found_keywords(text)
    
    if score >= 3:
        return True, f"✅ Valid logistics document. Found: {', '.join(found)}"
    else:
        return False, "⚠️ This document doesn't appear to be a logistics document."


# ============================================================
#  IMPROVED DOCUMENT TYPE DETECTION
# ============================================================

def detect_logistics_type(text):
    """Detect specific type of logistics document"""
    
    text_lower = text.lower()
    
    # Check each type with multiple keywords
    if any(x in text_lower for x in ['bill of lading', 'b/l no', 'bol no', 'sea waybill']):
        return "📦 Bill of Lading (B/L)"
    
    elif any(x in text_lower for x in ['airway bill', 'awb no', 'air waybill']):
        return "✈️ Airway Bill (AWB)"
    
    elif any(x in text_lower for x in ['lorry receipt', 'truck receipt', 'lr no']):
        return "🚛 Lorry Receipt"
    
    elif any(x in text_lower for x in ['commercial invoice', 'tax invoice', 'tax-invoice', 'inv no']):
        return "🧾 Commercial Invoice"
    
    elif any(x in text_lower for x in ['packing list', 'packing slip', 'pack list']):
        return "📋 Packing List"
    
    elif any(x in text_lower for x in ['purchase order', 'po no', 'p.o. no', 'order no']):
        return "📝 Purchase Order"
    
    elif any(x in text_lower for x in ['delivery note', 'delivery receipt', 'goods receipt', 'dn no']):
        return "🚚 Delivery Note"
    
    elif any(x in text_lower for x in ['customs declaration', 'bill of entry', 'customs clearance']):
        return "🛃 Customs Document"
    
    elif any(x in text_lower for x in ['warehouse receipt', 'warehousing', 'storage receipt']):
        return "🏭 Warehouse Receipt"
    
    elif any(x in text_lower for x in ['freight invoice', 'freight charges', 'shipping charges']):
        return "💰 Freight Invoice"
    
    elif any(x in text_lower for x in ['insurance', 'cargo insurance', 'marine insurance']):
        return "🛡️ Insurance Document"
    
    elif any(x in text_lower for x in ['certificate of origin', 'c/o certificate']):
        return "📜 Certificate of Origin"
    
    elif any(x in text_lower for x in ['manifest', 'shipping manifest']):
        return "📋 Shipping Manifest"
    
    else:
        return "📄 Logistics Document"


# ============================================================
#  MAIN FUNCTIONS (Unchanged)
# ============================================================

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def process_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )
    
    if not os.path.exists("faiss_index"):
        return "⚠️ No documents found. Please upload and 'Process' your PDFs first!"

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5) 
        
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
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]