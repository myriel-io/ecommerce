# 🛍️ H&M Fashion Search

This is an AI-powered fashion search engine that understands what you're looking for, even when you describe it in natural language. You can do keyword search and also semantic similarity to find clothes that actually match your intent.

## 🚀 Getting Started

The easiest way to get everything running is with a single command:

```bash
./start.sh
```

This script handles everything automatically: setting up your Python environment, installing dependencies, loading sample fashion data, and starting the web server.

If you prefer to set things up manually, you can create a virtual environment, install the required packages, and start the application yourself:

```bash
# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all the required packages
pip install -r requirements.txt

# Start the fashion search application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Once the server is running, open your browser and visit http://localhost:8000/static/index.html to start searching for fashion items.

## 🔍 How the Search Works

The search interface is intuitive and powerful. You can type natural language queries like "blue shirt" and the AI will find shirts, polos, and t-shirts in blue tones. The system understands that these items are semantically related, even if they don't contain the exact words you searched for.

You can also filter by category by clicking buttons like "Menswear" or "Ladieswear" to narrow down your results. If you want to browse everything available, just leave the search box empty and click "All" to see all products with pagination.

## 🛠️ What's Under the Hood

The backend is built with FastAPI and uses LanceDB as a vector database for fast similarity searches. The entire application is surprisingly compact at just 125 lines of clean, readable code. For AI processing, it uses Sentence Transformers to convert text descriptions into numerical vectors that capture semantic meaning.

The frontend is a clean HTML/CSS/JavaScript interface that connects to the API endpoints. The sample dataset includes 67 diverse H&M fashion items with AI-generated embeddings, giving you a good feel for how the search behaves with real product data.

## 📁 Project Structure

The codebase is intentionally minimal and focused. The main application lives in `app.py` (125 lines) and handles all the API endpoints and search logic. The `load_sample_data.py` script (47 lines) creates the sample fashion database with AI embeddings. The `start.sh` script provides one-click setup, while the `static/` directory contains the web interface.

## 🔧 If Something Goes Wrong

If you see a "port in use" error, kill any existing server processes with `pkill -f uvicorn`. Missing package errors usually mean you need to run `pip install -r requirements.txt` from within your activated virtual environment. If the search returns no results, try running `python load_sample_data.py` to make sure the sample data is properly loaded.

## 📚 The Technology Stack

This application demonstrates modern Python web development using FastAPI as the web framework. LanceDB provides vector database capabilities for similarity search operations. Sentence Transformers handles the AI text embeddings using the CLIP ViT-B-32 multimodal model, which understands both text and images.

---

**Ready to explore AI-powered fashion search? Just run `./start.sh` and visit localhost:8000 to get started!** 🚀