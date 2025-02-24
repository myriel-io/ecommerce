from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from pydantic import BaseModel

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "h&m-mini"
EMBEDDING_MODEL = "clip-ViT-B-32"

# Initialize FastAPI, Qdrant client and SentenceTransformer model
app = FastAPI()
qdrant_client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)
sentence_transformer = SentenceTransformer(EMBEDDING_MODEL)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchResult(BaseModel):
    image_url: str
    prod_name: str

@app.get("/")
async def read_root():
    return {"message": "H&M Fashion Search API"}

@app.get("/search/{query}", response_model=List[SearchResult])
async def search_fashion_items(query: str, limit: int = 6):
    """Search for fashion items using semantic search."""
    try:
        query_embedding = sentence_transformer.encode(query)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        return [
            SearchResult(
                image_url=hit.payload.get('image_url'),
                prod_name=hit.payload.get('prod_name')
            ) 
            for hit in search_results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
