"""H&M Fashion Search - Simple Version"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import lancedb
from sentence_transformers import SentenceTransformer

# Configuration
LANCEDB_PATH = "./data"
TABLE_NAME = "hm_mini"
EMBEDDING_MODEL = "clip-ViT-B-32"

# Data models
class SearchResult(BaseModel):
    image_url: str
    prod_name: str
    detail_desc: str
    product_type_name: str
    index_group_name: str
    price: float
    article_id: str
    available: bool
    color: str
    size: str

# Initialize FastAPI
app = FastAPI(title="H&M Fashion Search")

# Add CORS and static files
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database and AI model
db = lancedb.connect(LANCEDB_PATH)
encoder = SentenceTransformer(EMBEDDING_MODEL)

try:
    table = db.open_table(TABLE_NAME)
except:
    # Table doesn't exist, will be created by data loader
    table = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "H&M Fashion Search"}

@app.get("/groups", response_model=List[str])
def get_groups():
    if not table:
        return []
    
    groups_df = table.search().select(["index_group_name"]).to_pandas()
    if groups_df.empty:
        return []
    
    unique_groups = groups_df['index_group_name'].dropna().unique()
    return sorted([str(g) for g in unique_groups if g])

@app.get("/search", response_model=List[SearchResult])
def search_products(
    query: Optional[str] = None,
    group: Optional[str] = None,
    item: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    if not table:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Build filters
        filters = []
        if group:
            filters.append(f"index_group_name = '{group}'")
        if item:
            filters.append(f"product_type_name = '{item}'")
        
        filter_str = " AND ".join(filters) if filters else None
        
        # Search
        if query:
            # Semantic search with query
            query_vector = encoder.encode(query).tolist()
            search_results = table.search(query_vector)
        else:
            # Browse all products
            search_results = table.search()
        
        # Apply filters and pagination
        if filter_str:
            search_results = search_results.where(filter_str)
        
        results_df = search_results.limit(limit).offset(offset).to_pandas()
        
        if results_df.empty:
            return []
        
        # Convert to response format
        results = []
        for _, row in results_df.iterrows():
            results.append(SearchResult(**{
                "image_url": row.get("image_url", ""),
                "prod_name": row.get("prod_name", ""),
                "detail_desc": row.get("detail_desc", ""),
                "product_type_name": row.get("product_type_name", ""),
                "index_group_name": row.get("index_group_name", ""),
                "price": row.get("price", 0.0),
                "article_id": row.get("article_id", ""),
                "available": row.get("available", True),
                "color": row.get("color", ""),
                "size": row.get("size", "")
            }))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
