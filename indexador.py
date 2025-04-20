from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 
load_dotenv()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

docs_folder = "docs"
all_texts = []

for file_name in os.listdir(docs_folder):
    full_path = os.path.join(docs_folder, file_name)

    if file_name.endswith(".pdf") or file_name.endswith(".docx"):
        loader = UnstructuredFileLoader(full_path)
    elif file_name.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(full_path)
    elif file_name.endswith(".mp4") or file_name.endswith(".mkv"):
        loader = UnstructuredVideoLoader(full_path, api_key=OPENAI_API_KEY)
    else:
        continue

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    all_texts.extend(texts)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(all_texts, embeddings)
vectorstore.save_local("db_empresa")
print("Base vetorial gerada com sucesso!")