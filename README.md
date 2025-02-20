# H&M Fashion Search

A semantic search application for H&M fashion items using Streamlit, Qdrant vector database, and CLIP embeddings.

## Description

This application provides a user-friendly interface to search through H&M fashion items using natural language queries. It leverages semantic search capabilities through CLIP embeddings and Qdrant vector database to find visually similar fashion items.

## Features

- Semantic search using natural language queries
- Visual results display with product images and names
- Responsive grid layout showing up to 6 results
- Real-time search functionality

## Prerequisites

- Python 3.7+
- Qdrant running locally or remotely
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd h-and-m-fashion-search
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Qdrant is running on localhost:6333 (or update the configuration in app.py accordingly)

## Configuration

The application uses the following default configuration (can be modified in app.py):
- Qdrant Host: localhost
- Qdrant Port: 6333
- Collection Name: hnm
- Embedding Model: clip-ViT-B-32

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your search query in the text input field and click "Search"

## Technical Details

The application uses:
- Streamlit for the web interface
- Qdrant for vector similarity search
- CLIP (ViT-B-32) for generating embeddings
- SentenceTransformer for text encoding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- H&M for the fashion dataset
- OpenAI for the CLIP model
- Qdrant for the vector database

