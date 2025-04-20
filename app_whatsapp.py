from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 
import os
load_dotenv()

app = Flask(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render define essa variável
    app.run(host="0.0.0.0", port=port)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY não encontrada. Verifique o .env ou variáveis de ambiente.")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()

    # Carrega base vetorial
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ[OPENAI_API_KEY])
    db = FAISS.load_local("db_empresa", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.environ[OPENAI_API_KEY]),
        chain_type="stuff",
        retriever=retriever
    )

    resposta = qa.run(incoming_msg)

    # Envia resposta via Twilio
    twilio_resp = MessagingResponse()
    twilio_resp.message(resposta)

    return str(twilio_resp)
