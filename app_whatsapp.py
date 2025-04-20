from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Carrega variáveis do .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY não encontrada. Verifique o .env ou ambiente Render.")

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local("db_empresa", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever
    )

    resposta = qa.run(incoming_msg)

    twilio_resp = MessagingResponse()
    twilio_resp.message(resposta)
    return str(twilio_resp)

# Rota opcional para debug via navegador
@app.route("/", methods=["GET"])
def home():
    return "✅ App WhatsApp IA rodando!"

# Rodar servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
