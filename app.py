#import atexit
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
#from PyPDF2 import PdfReader
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

file_path = os.path.join(os.getcwd(), "60LEADERSONAI.pdf")
#loader = PyPDFLoader("60LEADERSONAI.pdf")
loader = PyPDFLoader(file_path)
data = loader.load()

print()
print(data)
print("***********************************")
print()

print (f'You have {len(data)} document(s) in your data')
print()
print (f'There are {len(data[0].page_content)} characters in your document')
print()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
print(text_splitter)
print()

db_texts = text_splitter.split_documents(data)
print(db_texts)
print("***********************************")
print()
print (f'Now you have {len(db_texts)} documents')

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

#Pinecone账号：
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
#def generate_random_string(length):
#    letters = string.ascii_letters
#    random_string = ''.join(random.choice(letters) for _ in range(length))
#    return random_string
#random_string = generate_random_string(8)

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))      
random_string = generate_random_string(8)

index_name = PINECONE_INDEX_NAME
#namespace = random_string
namespace = "HF-GRADIO"

#def exit_handler():
#    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
#    index_namespace_to_delete = pinecone.Index(index_name=index_name)
#    index_namespace_to_delete.delete(delete_all=True, namespace=namespace)

#atexit.register(exit_handler)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
#vector_db = Pinecone.from_texts(db_texts, hf_embeddings, index_name=index_name, namespace=namespace)
vector_db = Pinecone.from_texts([t.page_content for t in db_texts], hf_embeddings, index_name=index_name, namespace=namespace)
#docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name, namespace=namespace)
print("***********************************")
print("Pinecone Vector/Embedding DB Ready.")
print()

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"min_length":100,
                                   "max_new_tokens":1024, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

chain = load_qa_chain(llm=llm, chain_type="stuff")

def run_chain(user_query):
    if user_query !="" and not user_query.strip().isspace() and not user_query.isspace():
      print("Your query:\n"+user_query)
      vector_db_from_index = Pinecone.from_existing_index(index_name, hf_embeddings, namespace=namespace)
      ss_results = vector_db_from_index.similarity_search(query=user_query, namespace=namespace, k=5)
      initial_ai_response = chain.run(input_documents=ss_results, question=user_query)
      temp_ai_response = initial_ai_response.partition('<|end|>')[0]
      final_ai_response = temp_ai_response.replace('\n', '')
      return final_ai_response
    else:
      print("Invalid inputs.")  

iface = gr.Interface(fn=run_chain, inputs="text", outputs="text", title="AI Response")
iface.launch()
