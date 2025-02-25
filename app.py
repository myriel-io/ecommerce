from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel
from urllib.parse import unquote

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

@app.get("/search", response_model=List[SearchResult])
async def search_fashion_items(query: str = "", group: str = "", limit: int = 20):
    """Search for fashion items using semantic search and/or group filter."""
    try:
        # Decode URL-encoded parameters and clean them
        query = unquote(query.strip())
        group = unquote(group.strip())

        print(f"Search request - Query: '{query}', Group: '{group}'")  # Debug log

        # Create group filter condition
        conditions = None
        if group:
            conditions = {
                "must": [
                    {
                        "key": "index_group_name",
                        "match": {"value": group}
                    }
                ]
            }

        # If only group filter is provided (no query)
        if group and not query:
            try:
                scroll_results = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=limit,
                    filter=conditions,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                return [
                    SearchResult(
                        image_url=hit.payload.get('image_url', ''),
                        prod_name=hit.payload.get('prod_name', 'Unknown Product')
                    ) 
                    for hit in scroll_results
                ]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Qdrant scroll error: {str(e)}")

        # If query is provided (with or without group)
        if query:
            try:
                query_vector = sentence_transformer.encode(query).tolist()
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=conditions if conditions else None,
                    with_payload=True
                )
                
                return [
                    SearchResult(
                        image_url=hit.payload.get('image_url', ''),
                        prod_name=hit.payload.get('prod_name', 'Unknown Product')
                    ) 
                    for hit in search_results
                ]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        # If neither query nor group is provided
        raise HTTPException(status_code=400, detail="Please provide a search query or group")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/groups", response_model=List[str])
async def get_groups():
    """Get list of unique index group names."""
    try:
        groups = set()
        offset = 0
        limit = 100
        
        while True:
            try:
                scroll_results = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=limit,
                    offset=offset,
                    with_payload=["index_group_name"]
                )[0]
                
                if not scroll_results:
                    break
                    
                for result in scroll_results:
                    if group := result.payload.get('index_group_name'):
                        groups.add(group)
                        
                offset += limit
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Qdrant scroll error: {str(e)}")
        
        return sorted(list(groups))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
