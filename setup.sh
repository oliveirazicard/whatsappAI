#!/bin/bash

echo "🔧 Criando ambiente virtual..."
python3 -m venv .venv
source .venv/bin/activate

echo "⬇️ Instalando pacotes principais..."
pip install --upgrade pip
pip install langchain openai faiss-cpu streamlit unstructured tiktoken

echo "🎥 Instalando suporte a vídeos (Whisper via API)..."
pip install ffmpeg-python pydub

echo "✅ Instalação concluída!"

echo ""
echo "📦 (Opcional) Para usar Whisper localmente, execute manualmente:"
echo "pip install git+https://github.com/openai/whisper.git"
echo "pip install torch"

echo ""
echo "⚠️ Certifique-se de ter o ffmpeg instalado no sistema:"
echo "  - macOS: brew install ffmpeg"
echo "  - Linux: sudo apt install ffmpeg"
echo "  - Windows: baixe em https://ffmpeg.org/download.html"