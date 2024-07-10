from flask import Flask, render_template, jsonify, request
from langchain_openai import OpenAIEmbeddings
# from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
# import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
 #Initialize vector index 
vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


llm=CTransformers(model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.2})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    # app.run(host="0.0.0.0", port= 8080, debug= True)
    app.run(debug= True)