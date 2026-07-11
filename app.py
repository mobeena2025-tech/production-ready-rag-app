import streamlit as st
import os
from core_rag import process_pdf, get_rag_chain

st.set_page_config(page_title="Production RAG Platform", page_icon="🤖", layout="wide")
st.title("📚 Enterprise RAG: Chat with Private Documents")
st.caption("Built with LangChain, ChromaDB, and Gemini 1.5 Flash")

with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key successfully initialized!", icon="✅")
    else:
        st.warning("Please enter your Gemini API key to proceed. (Get a free key at Google AI Studio)")

uploaded_file = st.file_uploader("Upload a resource document (PDF)", type="pdf")

if uploaded_file and api_key:
    temp_dir = "./temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document, creating embeddings, and indexing vector space..."):
        if "retriever" not in st.session_state:
            st.session_state.retriever = process_pdf(file_path)
            st.success("Document Vector Indexing Complete!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your uploaded document:"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        rag_chain = get_rag_chain(st.session_state.retriever)
        with st.chat_message("assistant"):
            with st.spinner("Searching document context..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif not api_key:
    st.info("👈 Please enter your API key in the sidebar configuration to begin.")
