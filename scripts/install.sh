#!/bin/bash
set -e
echo "🚀 Soflution LLM - Installation"
if ! command -v ollama &> /dev/null; then
    echo "❌ Install Ollama first: https://ollama.com"
    exit 1
fi
echo "📥 Pulling Gemma 4 31B (~20GB)..."
ollama pull gemma4:31b
echo "⚙️ Creating Soflution LLM..."
ollama create soflution-llm -f Modelfile
echo "✅ Done! Run: ollama run soflution-llm"
