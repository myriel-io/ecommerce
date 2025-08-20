"""H&M Fashion Search - Expanded Sample Data Loader with 65+ Items"""

import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

# Configuration
LANCEDB_PATH = "./data"
TABLE_NAME = "hm_mini"
EMBEDDING_MODEL = "clip-ViT-B-32"

def create_product_data():
    """Generate 65+ diverse fashion products across all categories"""
    products = []
    
    # MENSWEAR (20 items)
    menswear = [
        ("MS001", "Classic Cotton Shirt", "Comfortable cotton shirt for casual and formal occasions", "Shirt", 29.99, "Blue", "M"),
        ("MS002", "Premium Cotton T-Shirt", "Soft premium cotton t-shirt with modern fit", "T-Shirt", 19.99, "White", "L"),
        ("MS003", "Casual Polo Shirt", "Classic polo with ribbed collar and cotton blend", "Polo", 24.99, "Navy", "L"),
        ("MS004", "Long Sleeve Henley", "Comfortable henley with button placket", "Henley", 27.99, "Gray", "M"),
        ("MS005", "Denim Jeans Regular Fit", "Classic blue denim with comfortable stretch", "Jeans", 49.99, "Blue", "32/32"),
        ("MS006", "Knit Sweater", "Warm crew neck sweater for cold weather", "Sweater", 39.99, "Charcoal", "L"),
        ("MS007", "Formal Dress Shirt", "Crisp white shirt with French cuffs", "Dress Shirt", 45.99, "White", "M"),
        ("MS008", "Chino Pants", "Versatile chinos for business casual wear", "Chinos", 34.99, "Khaki", "32/30"),
        ("MS009", "Hoodie Sweatshirt", "Comfortable hoodie with kangaroo pocket", "Hoodie", 32.99, "Black", "L"),
        ("MS010", "Flannel Shirt", "Soft flannel with plaid pattern", "Flannel", 28.99, "Red Plaid", "M"),
        ("MS011", "Tank Top", "Lightweight cotton tank for summer", "Tank Top", 12.99, "White", "M"),
        ("MS012", "Bomber Jacket", "Stylish bomber with ribbed cuffs", "Jacket", 59.99, "Olive", "L"),
        ("MS013", "V-Neck T-Shirt", "Classic v-neck with soft cotton blend", "V-Neck", 17.99, "Gray", "M"),
        ("MS014", "Cargo Shorts", "Comfortable shorts with multiple pockets", "Shorts", 25.99, "Beige", "32"),
        ("MS015", "Cardigan Sweater", "Button-up cardigan for layering", "Cardigan", 42.99, "Navy", "L"),
        ("MS016", "Athletic Shorts", "Moisture-wicking shorts for sports", "Athletic Shorts", 22.99, "Black", "M"),
        ("MS017", "Turtleneck Sweater", "Warm turtleneck with fine knit", "Turtleneck", 36.99, "Cream", "M"),
        ("MS018", "Plaid Shirt", "Classic plaid button-down shirt", "Plaid Shirt", 31.99, "Blue Plaid", "L"),
        ("MS019", "Zip-Up Hoodie", "Full-zip hoodie with pockets", "Zip Hoodie", 38.99, "Gray", "L"),
        ("MS020", "Dress Pants", "Formal pants with tailored fit", "Dress Pants", 52.99, "Charcoal", "34/32"),
    ]
    
    for item in menswear:
        products.append({
            "article_id": item[0], "prod_name": item[1], "detail_desc": item[2],
            "product_type_name": item[3], "index_group_name": "Menswear",
            "price": item[4], "available": True, "color": item[5], "size": item[6],
            "image_url": f"https://placehold.co/300x400/4A90E2/FFFFFF?text={item[3].replace(' ', '+')}"
        })
    
    # LADIESWEAR (20 items)
    ladieswear = [
        ("LD001", "Elegant Summer Dress", "Beautiful summer dress with floral pattern", "Dress", 49.99, "Floral", "S"),
        ("LD002", "Silk Chiffon Blouse", "Elegant silk blouse with delicate draping", "Blouse", 59.99, "Pink", "M"),
        ("LD003", "Wrap Style Dress", "Flattering wrap dress with tie waist", "Wrap Dress", 64.99, "Navy", "M"),
        ("LD004", "Flowy Tunic Top", "Comfortable tunic with loose fit", "Tunic", 34.99, "White", "L"),
        ("LD005", "High-Waist Skinny Jeans", "Trendy high-waist jeans with stretch", "Jeans", 45.99, "Dark Blue", "28/30"),
        ("LD006", "Cashmere Sweater", "Luxurious cashmere with crew neck", "Sweater", 89.99, "Beige", "S"),
        ("LD007", "Maxi Dress", "Flowing maxi dress for special occasions", "Maxi Dress", 72.99, "Coral", "M"),
        ("LD008", "Button-Down Shirt", "Classic button-down for work or casual", "Button Down", 38.99, "Light Blue", "M"),
        ("LD009", "A-Line Skirt", "Versatile a-line skirt at knee length", "Skirt", 29.99, "Black", "S"),
        ("LD010", "Lightweight Cardigan", "Open-front cardigan for layering", "Cardigan", 44.99, "Gray", "L"),
        ("LD011", "Pencil Skirt", "Professional pencil skirt with stretch", "Pencil Skirt", 35.99, "Navy", "M"),
        ("LD012", "Off-Shoulder Top", "Trendy off-shoulder with elastic neckline", "Off Shoulder", 27.99, "White", "S"),
        ("LD013", "Palazzo Pants", "Comfortable wide-leg pants", "Palazzo", 41.99, "Black", "M"),
        ("LD014", "Blazer Jacket", "Structured blazer for professional looks", "Blazer", 79.99, "Charcoal", "M"),
        ("LD015", "Bohemian Top", "Boho-style top with intricate patterns", "Boho Top", 33.99, "Multicolor", "L"),
        ("LD016", "Midi Dress", "Versatile midi dress with belt", "Midi Dress", 55.99, "Burgundy", "S"),
        ("LD017", "Denim Jacket", "Classic denim jacket with button closure", "Denim Jacket", 49.99, "Blue", "M"),
        ("LD018", "Sleeveless Blouse", "Elegant sleeveless blouse for warm weather", "Sleeveless", 31.99, "Cream", "S"),
        ("LD019", "Leggings", "Comfortable stretch leggings", "Leggings", 19.99, "Black", "M"),
        ("LD020", "Kimono Style Top", "Flowy kimono with wide sleeves", "Kimono", 42.99, "Rose", "L"),
    ]
    
    for item in ladieswear:
        products.append({
            "article_id": item[0], "prod_name": item[1], "detail_desc": item[2],
            "product_type_name": item[3], "index_group_name": "Ladieswear",
            "price": item[4], "available": True, "color": item[5], "size": item[6],
            "image_url": f"https://placehold.co/300x400/FF69B4/FFFFFF?text={item[3].replace(' ', '+')}"
        })
    
    # SPORT (12 items)
    sport = [
        ("SP001", "Athletic Performance Jacket", "Lightweight jacket with moisture-wicking technology", "Sports Jacket", 79.99, "Black", "L"),
        ("SP002", "Performance Shorts", "Athletic shorts with compression and quick-dry", "Athletic Shorts", 34.99, "Navy", "M"),
        ("SP003", "Yoga Leggings", "High-waist yoga pants with four-way stretch", "Yoga Pants", 42.99, "Charcoal", "S"),
        ("SP004", "Training Tank Top", "Breathable tank for high-intensity training", "Training Tank", 24.99, "Coral", "M"),
        ("SP005", "Running Tights", "Compression tights with reflective details", "Running Tights", 54.99, "Black", "L"),
        ("SP006", "Sports Bra", "High-support bra with removable pads", "Sports Bra", 29.99, "Pink", "M"),
        ("SP007", "Windbreaker Jacket", "Lightweight jacket for outdoor running", "Windbreaker", 65.99, "Blue", "L"),
        ("SP008", "Sweatpants", "Comfortable pants with elastic waistband", "Sweatpants", 39.99, "Gray", "M"),
        ("SP009", "Moisture-Wicking T-Shirt", "Technical tee with moisture control", "Athletic Tee", 27.99, "White", "L"),
        ("SP010", "Cycling Shorts", "Padded shorts for long bike rides", "Cycling Shorts", 47.99, "Black", "M"),
        ("SP011", "Cross-Training Shoes", "Versatile shoes for gym workouts", "Training Shoes", 89.99, "White/Black", "9"),
        ("SP012", "Swimming Trunks", "Quick-dry swim shorts for pool and beach", "Swim Shorts", 24.99, "Blue", "M"),
    ]
    
    for item in sport:
        products.append({
            "article_id": item[0], "prod_name": item[1], "detail_desc": item[2],
            "product_type_name": item[3], "index_group_name": "Sport",
            "price": item[4], "available": True, "color": item[5], "size": item[6],
            "image_url": f"https://placehold.co/300x400/2F4F4F/FFFFFF?text={item[3].replace(' ', '+')}"
        })
    
    # BABY/CHILDREN (8 items)
    kids = [
        ("BC001", "Twirl Dress for Girls", "Fun dress perfect for active kids", "Kids Dress", 24.99, "Purple", "6-7Y"),
        ("BC002", "Boys Graphic T-Shirt", "Colorful tee with fun designs", "Kids Tee", 14.99, "Blue", "8-9Y"),
        ("BC003", "Baby Onesie", "Soft cotton onesie with snap closures", "Onesie", 12.99, "Pink", "6-12M"),
        ("BC004", "Kids Denim Overalls", "Cute overalls with adjustable straps", "Overalls", 32.99, "Denim", "4-5Y"),
        ("BC005", "Girls Tutu Skirt", "Playful tutu for dress-up and dancing", "Tutu", 18.99, "Pink", "3-4Y"),
        ("BC006", "Boys Hoodie", "Comfortable hoodie with cartoon design", "Kids Hoodie", 26.99, "Red", "5-6Y"),
        ("BC007", "Baby Sleepsuit", "Cozy sleepsuit with footies", "Sleepsuit", 16.99, "Yellow", "3-6M"),
        ("BC008", "Kids Cargo Pants", "Durable pants with multiple pockets", "Kids Cargo", 28.99, "Khaki", "7-8Y"),
    ]
    
    for item in kids:
        products.append({
            "article_id": item[0], "prod_name": item[1], "detail_desc": item[2],
            "product_type_name": item[3], "index_group_name": "Baby/Children",
            "price": item[4], "available": True, "color": item[5], "size": item[6],
            "image_url": f"https://placehold.co/300x400/9370DB/FFFFFF?text={item[3].replace(' ', '+')}"
        })
    
    # DIVIDED (7 items)
    divided = [
        ("DV001", "Trendy Crop Top", "Modern crop top with unique design", "Crop Top", 15.99, "Black", "S"),
        ("DV002", "Ripped Skinny Jeans", "Distressed jeans with trendy rips", "Ripped Jeans", 39.99, "Light Blue", "26/30"),
        ("DV003", "Oversized Band Tee", "Vintage-style band t-shirt", "Band Tee", 22.99, "Gray", "M"),
        ("DV004", "High-Waist Shorts", "Trendy denim shorts for summer", "High Waist Shorts", 27.99, "Blue", "S"),
        ("DV005", "Graphic Sweatshirt", "Cozy sweatshirt with bold graphics", "Graphic Sweat", 34.99, "Pink", "M"),
        ("DV006", "Mini Skirt", "Trendy mini with A-line silhouette", "Mini Skirt", 24.99, "Black", "XS"),
        ("DV007", "Mesh Top", "Edgy mesh top for layering", "Mesh Top", 18.99, "White", "S"),
    ]
    
    for item in divided:
        products.append({
            "article_id": item[0], "prod_name": item[1], "detail_desc": item[2],
            "product_type_name": item[3], "index_group_name": "Divided",
            "price": item[4], "available": True, "color": item[5], "size": item[6],
            "image_url": f"https://placehold.co/300x400/FFD700/000000?text={item[3].replace(' ', '+')}"
        })
    
    return products

def main():
    products = create_product_data()
    print(f"Loading expanded fashion dataset with {len(products)} items...")
    
    # Connect to database
    db = lancedb.connect(LANCEDB_PATH)
    
    # Load AI model
    print("Loading AI model for embeddings...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Generate embeddings
    print("Generating AI embeddings for all products...")
    df['vector'] = df['detail_desc'].apply(lambda x: encoder.encode(x).tolist())
    
    # Save to database
    try:
        table = db.create_table(TABLE_NAME, df, mode="overwrite")
        print(f"✅ Successfully loaded {len(df)} products into {TABLE_NAME}")
        
        # Show breakdown by category
        category_counts = df['index_group_name'].value_counts()
        print("\n📊 Items by category:")
        for category, count in category_counts.items():
            print(f"   {category}: {count} items")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
