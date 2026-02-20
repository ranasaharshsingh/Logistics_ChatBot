import streamlit as st
from src.rag_engine import get_pdf_text, get_text_chunks, create_vector_store, process_query

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
    st.title("🚚 Logistics Document Chatbot")
    st.write("Upload your documents and start chatting.")

    # 4. UPLOADER SECTION (Inside an Expander to save space)
    with st.expander("📂 Upload & Sync Documents", expanded=True):
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        create_vector_store(text_chunks)
                        st.success("Sync Complete!")
                    else:
                        st.error("Could not read text from these PDFs.")
            else:
                st.error("Please upload files first.")

    # 5. CHAT INTERFACE
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question about your files..."):
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Thinking..."):
            response = process_query(user_question)
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()