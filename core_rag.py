import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def process_pdf(pdf_path: str):
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Chunk text into segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Initialize free Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create local Vector Store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    system_prompt = (
        "You are an expert assistant specialized in analyzing uploaded documents.\n"
        "Use the following pieces of retrieved context to answer the question. "
        "If you do not know the answer based on the context, say exactly: 'I cannot find the answer in the uploaded document.' "
        "Do not try to make up or extrapolate information outside the context.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
