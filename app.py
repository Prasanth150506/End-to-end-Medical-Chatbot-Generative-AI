from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Make sure keys are available to LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "test"

# Connect to Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Use Cohere LLM
llm = ChatCohere(model="command-r")

# Define prompt
from src.prompt import system_prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Chain setup
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"] if "answer" in response else response
    print("Response:", answer)
    return str(answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
