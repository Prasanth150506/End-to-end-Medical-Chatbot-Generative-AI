# End-to-end-Medical-Chatbot-Generative-AI

# Medibot — End-to-End Medical Chatbot

A Flask-powered medical chatbot that processes health queries using Hugging Face embeddings, Pinecone vector search, and an LLM (e.g., Cohere). Built with LangChain for easy RAG workflow.

---

##  Features

- Uses Hugging Face’s sentence-transformers for embeddings
- Stores and searches embeddings in Pinecone
- LLM integration via Cohere (or OpenAI if available)
- Simple web interface using Flask and Bootstrap
- Modular project structure for ease of development

---

##  Quick Start

### 1. Clone Project

```bash
git clone https://github.com/YourUsername/End-to-end-Medical-Chatbot-Generative.git
cd End-to-end-Medical-Chatbot-Generative


2. Create & Activate Environment
conda create --name medibot python=3.10 -y
conda activate medibot

3. Install Dependencies
pip install -r requirements.txt

4. Set Environment Variables

Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_key
COHERE_API_KEY=your_cohere_key        # for Cohere LLM
# or
OPENAI_API_KEY=your_openai_key       # if using OpenAI



5. Build the Pinecone Index
python store_index.py

6. Launch the Flask App
python app.py

open up localhost8080