from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.prompt import system_prompt
import os


app = Flask(__name__)


load_dotenv()

api_key = os.getenv("groq_api_key")
os.environ["GROQ_API_KEY"] = api_key
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY="a"

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

embedding=download_embeddings()

index_name="quickstart"

docsearch=PineconeVectorStore.from_existing_index(
    
    embedding=embedding,
    index_name=index_name
)

model="llama-3.1-8b-instant"
groq_model=ChatGroq(model=model)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human","{input}"),   
])

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
question_answer_chain=create_stuff_documents_chain(groq_model,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)








@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)