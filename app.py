import streamlit as st
# Import functions from our backend file
from src.rag_engine import get_pdf_text, get_text_chunks, create_vector_store, process_query

def main():
    st.set_page_config(page_title="Logistics AI", page_icon="🚚")

    st.title("🚚 Logistics Document Chatbot")
    st.write("Upload your invoices, manuals, or shipping documents and ask questions.")

    # SIDEBAR: For uploading files
    with st.sidebar:
        st.header("Upload Documents")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing... Please wait."):
                    # 1. Read PDF
                    raw_text = get_pdf_text(pdf_docs)
                    # 2. Split Text
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Create Memory
                    create_vector_store(text_chunks)
                    st.success("Done! You can now ask questions.")
            else:
                st.error("Please upload a file first.")

    # MAIN AREA: Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_question := st.chat_input("Ask a question about your files..."):
        # Show user message
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Get AI response
        with st.spinner("Thinking..."):
            response = process_query(user_question)
        
        # Show AI response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()