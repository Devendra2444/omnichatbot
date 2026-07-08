import os
import hashlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(verbose=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


def _data_hash() -> str:
    """Return a hash of all filenames and their sizes in the data dir.
    Used to detect if any files changed."""
    hasher = hashlib.md5()
    if not os.path.isdir(DATA_DIR):
        return ""
    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            hasher.update(f.encode())
            hasher.update(str(os.path.getmtime(path)).encode())
    return hasher.hexdigest()


@cl.cache
def load_and_index_documents():
    documents = []
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        extension = os.path.splitext(file)[1].lower()
        if extension not in LOADERS:
            continue
        loader = LOADERS[extension](file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_documents(
        embedding=embeddings,
        documents=chunks,
        persist_directory=DB_DIR,
    )
    return vector_store.as_retriever()


def load_cached_index():
    """Rebuild only if cached hash differs from current data hash."""
    hash_path = os.path.join(DB_DIR, ".data_hash")
    current_hash = _data_hash()

    if os.path.exists(DB_DIR) and os.path.exists(hash_path):
        with open(hash_path) as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=DB_DIR,
            )
            return vector_store.as_retriever()

    retriever = load_and_index_documents()
    os.makedirs(DB_DIR, exist_ok=True)
    with open(hash_path, "w") as f:
        f.write(current_hash)
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@cl.on_chat_start
async def start():
    msg = cl.Message(content="Loading and indexing documents...")
    await msg.send()

    retriever = load_cached_index()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful Multi-Agent AI Assistant.

    Use the provided context as your primary source.
    If the context doesn't fully answer the question,
    supplement with your own knowledge.

    Context:
    {context}

    Question:
    {question}
    """)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("chain", chain)
    await cl.Message(content="Ready! Ask me anything about the documents in the data folder.").send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")
    await msg.send()

    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk)

    await msg.update()
