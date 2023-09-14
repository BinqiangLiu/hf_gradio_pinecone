import gradio as gr
#from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import requests
import sys
#from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.chains.question_answering import load_qa_chain
#from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader
#from langchain.document_loaders import TextLoader
#from sentence_transformers.util import semantic_search
from pathlib import Path
from time import sleep
#import pandas as pd
#import torch
import os
import random
import string
from dotenv import load_dotenv
load_dotenv()

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))      
random_string = generate_random_string(10)

file_path = os.path.join(os.getcwd(), "valuation.pdf")
#loader = PyPDFLoader("60LEADERSONAI.pdf")
#loader = PyPDFLoader(file_path)
#data = loader.load()
#db_texts = text_splitter.split_documents(data)

data = PdfReader(file_path)
raw_text = ''
db_texts=''
for i, page in enumerate(data.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        text_splitter = RecursiveCharacterTextSplitter(        
#            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 100, #striding over the text
            length_function = len,
        )
        db_texts = text_splitter.split_text(raw_text)

class HFEmbeddings:
    def __init__(self, api_url, headers):
        self.api_url = api_url
        self.headers = headers
    def get_embeddings(self, texts):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        embeddings = response.json()
        return embeddings
    def embed_documents(self, texts):
        embeddings = self.get_embeddings(texts)
        return embeddings
    def __call__(self, texts):
        return self.embed_documents(texts)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_id = os.getenv('model_id')
hf_token = os.getenv('hf_token')
repo_id = os.getenv('repo_id')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

hf_embeddings = HFEmbeddings(api_url, headers)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
print(PINECONE_INDEX_NAME)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = PINECONE_INDEX_NAME
print(index_name)
namespace = random_string
print(namespace)

vector_db = Pinecone.from_texts(db_texts, hf_embeddings, index_name=index_name, namespace=namespace)
print("Pinecone Vector/Embedding DB Ready.")

index_name_extracted=pinecone.list_indexes()
print(index_name_extracted)

index_current = pinecone.Index(index_name=index_name)
index_status=index_current.describe_index_stats() 
print(index_status)

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

prompt_template = """You are a very helpful AI assistant. Please ONLY use the given context to answer the user's input question. If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}
Helpful AI Repsonse:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

def run_chain(user_query):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if user_query !="" and not user_query.strip().isspace() and not user_query.isspace():
      print("Your query:\n"+user_query)
      vector_db_from_index = Pinecone.from_existing_index(index_name, hf_embeddings, namespace=namespace)
      ss_results = vector_db_from_index.similarity_search(query=user_query, namespace=namespace, k=5)
      initial_ai_response = chain.run(input_documents=ss_results, question=user_query, return_only_outputs=True)        
      #initial_ai_response=chain({"input_documents": ss_results, "question": user_query}, return_only_outputs=True)            
      temp_ai_response = initial_ai_response.partition('<|end|>')[0]
      final_ai_response = temp_ai_response.replace('\n', '')
      print(final_ai_response)
      print(index_status)
      print(index_name_extracted)
      print(namespace) 
      print("****************")
      return final_ai_response
    else:
      print("Invalid inputs.")  

def delete_index_namespace():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_namespace_to_delete = pinecone.Index(index_name=index_name)
    index_namespace_to_delete.delete(delete_all=True, namespace=namespace)
    print("Pinecone Index Namespace: "+namespace+" has been deleted!")
        
with gr.Blocks() as demo:
    gr.Markdown("Enter your question below & click Get AI Response. Remember to clear data before exiting program.")
    with gr.Row():
        user_query = gr.Textbox(label="User query input box", placeholder="Enter your query here.")
        ai_response = gr.Textbox(label="AI Response display area", placeholder="AI Response to be displayed here.")
    query_btn = gr.Button("Get AI Response")
    ai_res_btn = gr.Button("Clear Data & Exit")
    query_btn.click(fn=run_chain, inputs=user_query, outputs=ai_response)
    ai_res_btn.click(fn=delete_index_namespace)

demo.launch()
