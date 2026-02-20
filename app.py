import streamlit as st
from PyPDF2 import PdfReader
from src.rag_engine import (
    get_pdf_text, 
    get_text_chunks, 
    create_vector_store, 
    process_query,
    validate_logistics_document_ai,
    detect_logistics_type
)

def main():
    # 1. Page Config - This MUST be the first Streamlit command
    st.set_page_config(page_title="Logistics AI", page_icon="🚚", layout="centered")

    # 2. Theme-Safe CSS (Fixes the white page issue)
    st.markdown("""
        <style>
        /* This ensures the main area is visible and respects themes */
        .main .block-container {
            padding-top: 2rem;
            max-width: 800px;
        }
        /* Style the chat messages without forcing a white background */
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. HEADER
    st.title(" Logistics Document Chatbot")
    st.write("Upload your logistics documents and start chatting.")

    # 4. UPLOADER SECTION (Inside an Expander to save space)
    with st.expander("📂 Upload & Sync Documents", expanded=True):
        pdf_docs = st.file_uploader(
            "Choose PDF files (Invoices, Bills of Lading, Shipping Docs)", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process Documents", use_container_width=True):
            if pdf_docs:
                # Step 1: Validate each document
                valid_docs = []
                invalid_files = []
                valid_doc_types = []
                
                with st.spinner("Validating documents... ⏳"):
                    
                    for pdf in pdf_docs:
                        # Extract text from PDF
                        pdf_reader = PdfReader(pdf)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        
                        # Check if text is empty
                        if not text or len(text.strip()) < 50:
                            invalid_files.append((pdf.name, "⚠️ This document appears to be empty or unreadable."))
                            continue
                        
                        # Validate using AI
                        is_valid, message = validate_logistics_document_ai(text)
                        
                        if is_valid:
                            valid_docs.append(pdf)
                            # Detect document type
                            doc_type = detect_logistics_type(text)
                            valid_doc_types.append((pdf.name, doc_type))
                        else:
                            invalid_files.append((pdf.name, message))
                
                # Step 2: Show validation results
                if invalid_files:
                    st.error("❌ **Some files were rejected:**")
                    for filename, msg in invalid_files:
                        st.write(f"📄 **{filename}**: {msg}")
                    st.warning("💡 Please upload only logistics-related documents (Invoices, Bills of Lading, Shipping Documents, Packing Lists, etc.)")
                    st.divider()
                
                # Step 3: Process valid documents
                if valid_docs:
                    with st.spinner("Processing logistics documents... 📄"):
                        raw_text = get_pdf_text(valid_docs)
                        
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            create_vector_store(text_chunks)
                            
                            # Show success message with document types
                            st.success(f"✅ {len(valid_docs)} document(s) processed successfully!")
                            
                            # Show what types of documents were uploaded
                            st.info("📋 **Uploaded Document Types:**")
                            for filename, doc_type in valid_doc_types:
                                st.write(f"   • {filename}: {doc_type}")
                        else:
                            st.error("Could not read text from these PDFs.")
                elif not invalid_files:
                    st.warning("⚠️ No valid logistics documents found. Please upload logistics-related PDFs.")
                        
            else:
                st.error("Please upload files first.")

    # 5. CHAT INTERFACE
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about your logistics documents..."):
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Thinking... 💭"):
            response = process_query(user_question)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()