from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
import lancedb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pydantic import BaseModel
from urllib.parse import unquote
import numpy as np
import pandas as pd
import base64

# Configuration
class Config:
    LANCEDB_PATH = "./data"
    TABLE_NAME = "hm_mini"
    EMBEDDING_MODEL = "clip-ViT-B-32"
    GROUP_ORDER = ["Menswear", "Ladieswear", "Divided", "Baby/Children", "Sport"]

class SearchResult(BaseModel):
    image_url: str
    prod_name: str
    detail_desc: str
    product_type_name: str
    index_group_name: str
    price: float = 0.0
    article_id: str = ""
    available: bool = True
    color: str = ""
    size: str = ""

    class Config:
        from_attributes = True

class LanceDBService:
    def __init__(self):
        self.db = lancedb.connect(Config.LANCEDB_PATH)
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.table = None
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure the table exists, create if it doesn't"""
        try:
            self.table = self.db.open_table(Config.TABLE_NAME)
        except:
            # Table doesn't exist, we'll need to create it
            # For now, we'll create an empty table with the expected schema
            schema = {
                "image_url": "string",
                "image_data": "string",  # Base64 encoded image data
                "prod_name": "string", 
                "detail_desc": "string",
                "product_type_name": "string",
                "index_group_name": "string",
                "price": "float64",
                "article_id": "string",
                "available": "bool",
                "color": "string",
                "size": "string",
                "vector": "float32[512]"  # Assuming 512-dimensional vectors
            }
            self.table = self.db.create_table(Config.TABLE_NAME, schema=schema)

    def create_filter(self, groups: List[str] = None, items: List[str] = None) -> str:
        """Create LanceDB filter string"""
        conditions = []
        if groups:
            group_condition = " OR ".join([f"index_group_name = '{group}'" for group in groups])
            conditions.append(f"({group_condition})")
        if items:
            item_condition = " OR ".join([f"product_type_name = '{item}'" for item in items])
            conditions.append(f"({item_condition})")
        
        if conditions:
            return " AND ".join(conditions)
        return None

    async def search(self, query: str, groups: List[str], items: List[str], 
                    limit: int, offset: int) -> List[SearchResult]:
        try:
            filter_condition = self.create_filter(groups, items)
            
            if not query:
                # No query, just filter and paginate
                if filter_condition:
                    results = self.table.search().where(filter_condition).limit(limit).to_pandas()
                else:
                    results = self.table.search().limit(limit).to_pandas()
            else:
                # Semantic search with query
                query_vector = self.encoder.encode(query).tolist()
                
                search_query = self.table.search(query_vector)
                if filter_condition:
                    search_query = search_query.where(filter_condition)
                
                results = search_query.limit(limit).to_pandas()

            if results.empty:
                return []

            # Convert DataFrame to list of SearchResult objects
            search_results = []
            for _, row in results.iterrows():
                search_results.append(SearchResult(
                    image_url=row.get('image_url', ''),
                    prod_name=row.get('prod_name', 'Unknown Product'),
                    detail_desc=row.get('detail_desc', 'No description available'),
                    product_type_name=row.get('product_type_name', ''),
                    index_group_name=row.get('index_group_name', ''),
                    price=float(row.get('price', 0.0)),
                    article_id=row.get('article_id', ''),
                    available=row.get('available', True),
                    color=row.get('color', ''),
                    size=row.get('size', '')
                ))
            
            return search_results

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Search error: {str(e)}"
            )

    async def get_groups(self) -> List[str]:
        try:
            # Get unique groups from the table
            groups_df = self.table.search().select(["index_group_name"]).to_pandas()
            
            if groups_df.empty:
                return []
            
            # Get unique values and filter out None/NaN
            groups = set()
            for group in groups_df['index_group_name'].dropna().unique():
                if group and str(group).strip():
                    groups.add(str(group).strip())
            
            # Sort according to GROUP_ORDER preference
            ordered_groups = [g for g in Config.GROUP_ORDER if g in groups]
            remaining_groups = sorted([g for g in groups if g not in Config.GROUP_ORDER])
            
            return ordered_groups + remaining_groups
            
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch groups: {str(e)}"
            )

# Initialize FastAPI and services
app = FastAPI(title="H&M Fashion Search API")
lancedb_service = LanceDBService()

# Configure CORS and static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/search", response_model=List[SearchResult])
async def search_fashion_items(
    query: str = "", 
    group: List[str] = Query(default=[]),
    item: List[str] = Query(default=[]),
    limit: int = 20
):
    """Search for fashion items using semantic search and/or filters."""
    query = unquote(query.strip())
    groups = [unquote(g.strip()) for g in group]
    items = [unquote(i.strip()) for i in item]
    return await lancedb_service.search(query, groups, items, limit, 0)

@app.get("/groups", response_model=List[str])
async def get_groups():
    """Get list of unique index group names in specified order."""
    return await lancedb_service.get_groups()

@app.get("/image/{article_id}")
async def get_image(article_id: str):
    """Serve binary image data from the database."""
    try:
        # Search for the item by article_id
        results = lancedb_service.table.search().where(f"article_id = '{article_id}'").limit(1).to_pandas()
        
        if results.empty:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_data = results.iloc[0]['image_data']
        
        # Decode base64 image data
        try:
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Determine content type based on image format
            if image_bytes.startswith(b'\xff\xd8\xff'):
                content_type = "image/jpeg"
            elif image_bytes.startswith(b'\x89PNG'):
                content_type = "image/png"
            elif image_bytes.startswith(b'GIF'):
                content_type = "image/gif"
            elif image_bytes.startswith(b'WEBP'):
                content_type = "image/webp"
            else:
                content_type = "image/jpeg"  # Default fallback
            
            return Response(content=image_bytes, media_type=content_type)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error decoding image: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")
