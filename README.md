# H&M Fashion Search

A semantic fashion search engine using FastAPI, Qdrant, and CLIP embeddings.

## Features

- 🔍 Semantic search with CLIP embeddings
- 🎯 Category and item type filtering
- 📱 Responsive web interface
- ⚡ Real-time search results
- 🔄 Infinite scroll pagination

## Quick Start

1. Clone and install:
```bash
git clone <repo-url>
cd h-and-m-fashion-search
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Start Qdrant (ensure it's running on localhost:6333)

3. Run the app:
```bash
uvicorn app:app --reload
```

4. Visit http://localhost:8000/static/index.html

## API

- GET /search - Search with filters
- GET /groups - Get product groups
- API docs: http://localhost:8000/docs

## Tech Stack

- Backend: FastAPI, Qdrant, CLIP
- Frontend: HTML, CSS, JavaScript
- Database: Qdrant Vector DB

## License

[Your License]

