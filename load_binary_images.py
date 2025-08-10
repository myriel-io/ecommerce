#!/usr/bin/env python3
"""
Binary image data loader for H&M Fashion Search with LanceDB
This script downloads images and stores them as base64-encoded binary data in LanceDB.
"""

import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import time

def download_and_encode_image(url, max_size=(400, 400), quality=85):
    """Download an image and encode it as base64"""
    try:
        print(f"Downloading image: {url}")
        
        # Download the image
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Open and process the image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize if needed
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save as JPEG to BytesIO
        output = BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        # Encode as base64
        encoded = base64.b64encode(output.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
        
    except Exception as e:
        print(f"Error downloading/encoding image {url}: {e}")
        # Return a simple colored placeholder as base64
        return create_placeholder_image(max_size, color=(180, 180, 180))

def create_placeholder_image(size=(400, 400), color=(180, 180, 180)):
    """Create a simple placeholder image as base64"""
    try:
        img = Image.new('RGB', size, color)
        output = BytesIO()
        img.save(output, format='JPEG', quality=85)
        output.seek(0)
        encoded = base64.b64encode(output.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"Error creating placeholder: {e}")
        return ""

def create_sample_data_with_binary_images():
    """Create sample fashion data with binary image data"""
    
    # Sample image URLs from reliable sources
    image_urls = [
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=400&fit=crop",  # shirt
        "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=400&fit=crop",  # dress
        "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=400&fit=crop",   # jeans
        "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=400&fit=crop",   # sneakers
        "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=400&fit=crop",  # dress
        "https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?w=400&h=400&fit=crop",  # jacket
        "https://images.unsplash.com/photo-1617137984095-74e4e5e3613f?w=400&h=400&fit=crop",  # sweater
        "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=400&fit=crop",  # shoes
        "https://images.unsplash.com/photo-1562157873-818bc0726f68?w=400&h=400&fit=crop",   # t-shirt
        "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=400&h=400&fit=crop",  # shoes
    ]
    
    sample_data = [
        {
            "image_url": "binary_stored",  # Indicate that image is stored as binary
            "prod_name": "Classic Cotton Shirt",
            "detail_desc": "A comfortable and stylish cotton shirt perfect for casual and formal occasions. Features a modern fit with a classic collar design.",
            "product_type_name": "Shirt",
            "index_group_name": "Menswear",
            "price": 29.99,
            "article_id": "MS001",
            "available": True,
            "color": "Blue",
            "size": "M"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Elegant Summer Dress",
            "detail_desc": "Light and breezy summer dress with floral patterns. Made from breathable fabric perfect for warm weather.",
            "product_type_name": "Dress",
            "index_group_name": "Ladieswear",
            "price": 45.50,
            "article_id": "LD001",
            "available": True,
            "color": "Floral",
            "size": "S"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Slim Fit Jeans",
            "detail_desc": "Modern slim-fit jeans with stretch fabric for comfort and style. Perfect for everyday wear.",
            "product_type_name": "Jeans",
            "index_group_name": "Menswear",
            "price": 59.99,
            "article_id": "MJ001",
            "available": True,
            "color": "Dark Blue",
            "size": "L"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Athletic Sneakers",
            "detail_desc": "Comfortable athletic sneakers with advanced cushioning technology. Perfect for sports and casual wear.",
            "product_type_name": "Shoes",
            "index_group_name": "Sport",
            "price": 89.99,
            "article_id": "SS001",
            "available": True,
            "color": "White",
            "size": "42"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Business Dress",
            "detail_desc": "Professional business dress suitable for office wear. Features a tailored fit and premium fabric.",
            "product_type_name": "Dress",
            "index_group_name": "Ladieswear",
            "price": 79.99,
            "article_id": "LD002",
            "available": True,
            "color": "Black",
            "size": "M"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Winter Jacket",
            "detail_desc": "Warm winter jacket with insulation and water-resistant coating. Perfect for cold weather.",
            "product_type_name": "Jacket",
            "index_group_name": "Menswear",
            "price": 129.99,
            "article_id": "MJ002",
            "available": True,
            "color": "Navy",
            "size": "L"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Cozy Sweater",
            "detail_desc": "Soft and warm sweater made from premium wool blend. Perfect for layering in cooler weather.",
            "product_type_name": "Sweater",
            "index_group_name": "Ladieswear",
            "price": 55.00,
            "article_id": "LS001",
            "available": True,
            "color": "Cream",
            "size": "S"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Casual Loafers",
            "detail_desc": "Elegant casual loafers made from genuine leather. Perfect for both casual and semi-formal occasions.",
            "product_type_name": "Shoes",
            "index_group_name": "Menswear",
            "price": 95.00,
            "article_id": "MS002",
            "available": True,
            "color": "Brown",
            "size": "43"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Basic T-Shirt",
            "detail_desc": "High-quality basic t-shirt made from 100% organic cotton. A wardrobe essential in various colors.",
            "product_type_name": "T-shirt",
            "index_group_name": "Divided",
            "price": 12.99,
            "article_id": "DT001",
            "available": True,
            "color": "White",
            "size": "M"
        },
        {
            "image_url": "binary_stored",
            "prod_name": "Running Shoes",
            "detail_desc": "High-performance running shoes with advanced sole technology and breathable upper material.",
            "product_type_name": "Shoes",
            "index_group_name": "Sport",
            "price": 119.99,
            "article_id": "SS002",
            "available": True,
            "color": "Black/Red",
            "size": "41"
        }
    ]
    
    # Download and encode images
    print("Downloading and encoding images...")
    for i, item in enumerate(sample_data):
        if i < len(image_urls):
            item["image_data"] = download_and_encode_image(image_urls[i])
        else:
            # Create placeholder for items without specific image URLs
            item["image_data"] = create_placeholder_image()
        
        # Small delay to be respectful to image servers
        time.sleep(0.5)
    
    return sample_data

def main():
    # Connect to LanceDB
    db = lancedb.connect("./data")
    table_name = "hm_mini"
    
    print(f"Creating sample data with binary images...")
    sample_data = create_sample_data_with_binary_images()
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Load the sentence transformer model for generating embeddings
    print("Loading sentence transformer model...")
    model = SentenceTransformer('clip-ViT-B-32')
    
    # Generate embeddings from product descriptions
    print("Generating embeddings...")
    descriptions = df['detail_desc'].tolist()
    embeddings = model.encode(descriptions)
    
    # Add embeddings to the DataFrame
    df['vector'] = embeddings.tolist()
    
    # Drop existing table if it exists and create new one
    try:
        db.drop_table(table_name)
        print(f"Dropped existing table: {table_name}")
    except Exception as e:
        print(f"Table {table_name} doesn't exist or couldn't be dropped: {e}")
    
    # Create new table with binary image data
    print(f"Creating table {table_name} with {len(df)} items...")
    table = db.create_table(table_name, df)
    
    print(f"✅ Successfully created table '{table_name}' with {len(df)} items and binary image data!")
    print("\nSample article IDs for testing:")
    for i, row in df.iterrows():
        print(f"  - {row['article_id']}: {row['prod_name']}")
    
    print(f"\n🖼️ Images are now stored as binary data in the database!")
    print(f"📡 You can access images via: http://localhost:8000/image/<article_id>")
    print(f"🌐 Example: http://localhost:8000/image/MS001")

if __name__ == "__main__":
    main()
