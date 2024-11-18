Assignment 1

FastAPI Application
Overview
    This FastAPI application provides endpoints for processing URLs, uploading PDFs, and handling chat interactions. It uses BERT for text embeddings and similarity calculations to provide relevant responses based on the content stored.

Features
    Process URL: Extract text content from a given URL and store it.
    Upload PDF: Upload a PDF file, extract its text, and store it.
    Chat: Query stored content based on a chat ID and receive the most relevant response.
    Get Stored Data: Retrieve all stored data including URLs and PDF contents.

Getting Started

Prerequisites
  Docker
  Python

Running the Application
    1.  Using Docker
            Build the Docker Image
                docker build -t fastapi-app .

            Run the Docker Container
                docker run -p 80:80 fastapi-app
    
    2.  Without Docker
            Run the FastAPI Application
                uvicorn main:app --host 0.0.0.0 --port 80


API Endpoints
    POST /process_url
    Description: Extracts and stores text from a given URL.
    Request Body: { "url": "string" }
    Response: { "chat_id": "string", "message": "string" }

    POST /process_pdf
    Description: Uploads and processes a PDF file.
    Form Data: file: <pdf file>
    Response: { "chat_id": "string", "message": "string" }

    POST /chat
    Description: Queries stored content based on a chat ID and question.
    Request Body: { "chat_id": "string", "question": "string" }
    Response: { "response": "string" }

    GET /get_data
    Description: Retrieves all stored data.
    Response: { "data": "object" }

UI Access URL
    http://127.0.0.1:8000/static/index.html

Acknowledgments
    FastAPI: The modern, fast (high-performance) web framework for building APIs with Python.
    BERT: Pre-trained Transformer model for various NLP tasks.
    Docker`: Containerization platform
