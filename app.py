import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 
load_dotenv()

import os

st.set_page_config(page_title="Chat Corporativo", layout="centered")
st.title("🤖 Chat Corporativo - PoC")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY não encontrada. Verifique o .env ou variáveis de ambiente.")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
query = st.text_input("Digite sua pergunta sobre os documentos da empresa:")

if query:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local("db_empresa", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever
    )

    result = qa.run(query)
    st.markdown(f"**Resposta:** {result}")