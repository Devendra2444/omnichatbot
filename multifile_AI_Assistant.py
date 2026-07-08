import os
from os import getenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv(verbose=True)
path = "/home/deven/Projects/RAG/OmniRAG/data"
documents = []

'''if os.path.exists(path):
    for file in os.listdir(path):

        file_path = os.path.join(path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            continue

        documents.extend(loader.load())

else:
    raise FileNotFoundError("File doesn't exist!")
'''
# OR
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}

if not os.path.isdir(path):
    raise FileNotFoundError(f"Directory not found: {path}")

for file in os.listdir(path):

    file_path = os.path.join(path, file)

    extension = os.path.splitext(file)[1].lower()

    if extension not in LOADERS:
        continue

    loader = LOADERS[extension](file_path)

    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages/documents.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
    embedding=embeddings,
    documents=chunks,
)

retriever = vector_store.as_retriever()

llm = ChatGroq(

    model="llama-3.3-70b-versatile",
    temperature=0.3,
)

# prompt = ChatPromptTemplate.from_template(
#     """
#     You are multiAgent AI Assistant.
#
#     Context:{context}
#
#     Question:{question}
#
#
#     """
# )

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful Multi-Agent AI Assistant.
    
    Use ONLY the provided content to answer the user's question.
    
    If the answer is not present in the content, say:
    "I couldn't find the answer in the provided documents."
    
    Context:
    {context}
    
    Question:
    {question}
    """
)

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

"""input = {"content": retreiver | format_docs,
    "question":RunnablePassthrough()}"""

chain = (
    {"context": retriever | format_docs,
    "question":RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)

query = input("Ask:")
ans = chain.invoke(query)

print("\nAnswer:\n")
print(ans)

