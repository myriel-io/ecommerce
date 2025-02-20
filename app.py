import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_HOST = "localhost"  # Change if your Qdrant instance is hosted elsewhere
QDRANT_PORT = 6333
COLLECTION_NAME = "hnm"
EMBEDDING_MODEL = "clip-ViT-B-32"

# Initialize Qdrant client and SentenceTransformer model
qdrant_client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT)
sentence_transformer = SentenceTransformer(EMBEDDING_MODEL)

def search_fashion_items(query: str, limit: int = 6):
    """Search for fashion items using semantic search."""
    query_embedding = sentence_transformer.encode(query)
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    return [(hit.payload.get('image_url'), hit.payload.get('prod_name')) for hit in search_results]

def main():
    st.title("H&M Fashion Search")
    st.write("Enter a search query to find fashion items in the H&M collection.")

    query = st.text_input("Search Query", "")
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_fashion_items(query)
                if results:
                    st.subheader("Search Results:")
                    
                    # Create three rows with two columns each
                    for row in range(0, 6, 2):
                        cols = st.columns(2)
                        for col in range(2):
                            if row + col < len(results):
                                with cols[col]:
                                    image_url, prod_name = results[row + col]
                                    st.write(f"**{prod_name}**")
                                    try:
                                        st.image(image_url, width=300)
                                    except Exception as e:
                                        st.error(f"Could not load image: {image_url}")
                else:
                    st.warning("No matching items found.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
