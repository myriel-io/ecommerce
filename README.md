# H&M Fashion Search

A semantic search application for H&M fashion items using FastAPI, Qdrant vector database, and CLIP embeddings.

## Description

This application provides a REST API and web interface to search through H&M fashion items using natural language queries. It leverages semantic search capabilities through CLIP embeddings and Qdrant vector database to find visually similar fashion items.

## Features

- RESTful API for semantic search
- Web interface with responsive design
- Visual results display with product images and names
- Real-time search functionality
- CORS enabled for API access

## Prerequisites

- Python 3.11+
- Qdrant running locally or remotely
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd h-and-m-fashion-search
```

2. Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure Qdrant is running on localhost:6333 (or update the configuration in app.py accordingly)

## Configuration

The application uses the following default configuration (can be modified in app.py):
- Qdrant Host: localhost
- Qdrant Port: 6333
- Collection Name: h&m-mini
- Embedding Model: clip-ViT-B-32

## Usage

1. Start the FastAPI application:

```bash
uvicorn app:app --reload
```

2. Access the application:
- Web Interface: http://localhost:8000/static/index.html
- API Documentation: http://localhost:8000/docs
- API Base URL: http://localhost:8000

## API Endpoints

- `GET /`: Health check endpoint
- `GET /search/{query}`: Search endpoint that accepts a query string and returns matching fashion items
- Static files served from `/static` directory

## Technical Details

The application uses:
- FastAPI for the REST API
- Qdrant for vector similarity search
- CLIP (ViT-B-32) for generating embeddings
- SentenceTransformer for text encoding
- HTML/JavaScript for the frontend interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- H&M for the fashion dataset
- OpenAI for the CLIP model
- Qdrant for the vector database

