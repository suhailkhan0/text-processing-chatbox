from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
from bs4 import BeautifulSoup
import pdfplumber
import uuid
import os, sys
import json
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the directory where index.html is located
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


DATA_FILE = 'data/storage.json'
UPLOAD_DIR = 'data/uploads/'

if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump({}, f)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def read_data(): #reads .json and returns a dict 
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def write_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    chat_id: str
    question: str

def extract_text_from_url(url):
    """Extracts the main text content from a webpage."""
    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch the URL content")
    
    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the paragraph tags and concatenate their text
    content = ''
    paragraphs = soup.find_all('p')
    for para in paragraphs:
        content += '{}\n'.format(para.get_text())
    
    
    return content


def generate_embeddings_bert(texts, tokenizer, model):
    """Generate embeddings using BERT."""
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the mean of the last hidden state for the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file.
    """
    pdf_text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text+' '
    #pdf_text = ' '.join(pdf_text.split())
    return pdf_text

def find_relevant_content(storage, query, content):
    """
    Finds the most relevant content by computing cosine similarity between the query
    and stored content embeddings.
    """
    content_sections = content.split('\n')
    print(content_sections, len(content_sections))
    
    # Generate embeddings for content sections
    content_embeddings = generate_embeddings_bert(content_sections, tokenizer, model)
    splits = [content_sections[0]]
    current_chunk = content_sections[0]
    
    for i in range(1, len(content_sections)):
        similarity = np.dot(content_embeddings[i], content_embeddings[i-1]) / (np.linalg.norm(content_embeddings[i]) * np.linalg.norm(content_embeddings[i-1]))
        if similarity < 0.75:
            splits.append(content_sections[i])
        else:
            current_chunk += ' ' + content_sections[i]
            splits[-1] = current_chunk
    content_sections = splits
    content_embeddings = generate_embeddings_bert(content_sections, tokenizer, model)
    print(content_sections, len(content_sections))

    # Generate embedding for the query
    query_embedding = generate_embeddings_bert([query], tokenizer, model)[0]
    
    # Find the most relevant section
    similarities = cosine_similarity([query_embedding], content_embeddings)
    most_relevant_index = np.argmax(similarities)
    print(similarities)
    print(most_relevant_index)
    similarity_score = similarities[0][most_relevant_index]
    print(similarity_score)
    
    # Return the most relevant response
    return content_sections[most_relevant_index]
    
    raise HTTPException(status_code=500, detail="Content not found")


@app.post("/app/remove_pdf")
async def remove_pdf(chat_id: str):
    storage = read_data()
    if chat_id not in storage:
        raise HTTPException(status_code=404, detail="Chat ID not found")
    
    file_path = storage[chat_id].get('file_path')
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    
    del storage[chat_id]
    write_data(storage)
    
    return {"message": "PDF file removed successfully."}

@app.post("/process_url")
async def process_url(request: URLRequest):
    request: URLRequest
    url = request.url
    storage = read_data()
    for key in storage.keys():
        if 'url' in storage[key].keys() and storage[key]['url'] == url:
            return "url alrady exists for chat id "+  key
    text = extract_text_from_url(url)
    chat_id = str(uuid.uuid4())
    storage[chat_id] = {'content': text, 'type': 'url', 'url': url}
    write_data(storage)

    return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF document.")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(file)
    chat_id = str(uuid.uuid4())
    storage = read_data()
    for key in storage.keys():
        if 'file_path' in storage[key].keys() and storage[key]['file_path'] == file_path:
            return "pdf file alrady exists for chat id "+  key
    storage[chat_id] = {'content': pdf_text, 'type': 'pdf', 'file_path': file_path}
    write_data(storage)
    
    return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}

@app.post("/chat")
async def chat(request: ChatRequest):
    storage = read_data()
    if request.chat_id not in storage:
        raise HTTPException(status_code=404, detail="Chat ID not found")
    print(storage)
    stored_content = storage[request.chat_id]['content']
    print('content@@@@', stored_content)

    response = find_relevant_content(storage[request.chat_id], request.question, stored_content)

    return {"response": response}

@app.get("/get_data")
async def get_chat_ids():
    storage = read_data()
    return storage
