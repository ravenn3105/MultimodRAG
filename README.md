# Multimodal RAG with Hierarchical Chunking & Rate Limiting

Enhanced Streamlit application for Retrieval-Augmented Generation (RAG) that indexes text and images, preserves document structure through hierarchical chunking, and enforces client-side rate limiting for Cohere embeddings.

## Features

- **Multimodal Retrieval**: Index text and image content from PDFs.
- **Hierarchical Chunking**: Split documents into nested levels (sections, paragraphs, sentences) for improved context preservation.
- **Vector Search**: FAISS-based nearest-neighbor search for embeddings.
- **LLM Integration**: Generate answers using Google Gemini.
- **Rate Limiting**: Token-bucket implementation (90 calls/minute) to avoid 429 errors on Cohere Trial key.
- **Interactive UI**: Streamlit interface with three tabs:
  1. **Index Documents** – Upload and process PDFs with optional hierarchical chunking.
  2. **Search** – Perform hierarchical or standard search with real-time API quota display.
  3. **Hierarchical Analysis** – Visualize chunk distribution and explore document structure.
- **App Health Indicator**: “✅ App is running!” banner on startup, plus quota metrics.

## Quick Start

### Prerequisites

- Python 3.9 or higher  
- Cohere Trial API key  
- Google Gemini API key  

### Installation

1. Clone this repository:
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag


2. Install dependencies:
pip install -r requirements.txt


3. Create a `config.py` in the project root:
COHERE_API_KEY = "your-cohere-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-pro"


## Project Structure

multimodal-rag/
├── enhanced_app.py # Main Streamlit application
├── embeddings.py # Rate-limited Cohere embeddings
├── rate_limiter.py # Token-bucket rate limiter
├── hierarchical_chunker.py # Hierarchical chunking implementation
├── enhanced_document_utils.py # PDF parsing & chunk management
├── search.py # FAISS-based retrieval logic
├── config.py # API key configuration
├── requirements.txt # Python dependencies
└── data/ # Persisted FAISS index & metadata


## Usage

1. Run the app:
streamlit run enhanced_app.py

2. **Index Documents**  
- Navigate to the **Index Documents** tab.  
- Upload one or more PDF files.  
- Toggle **Use Hierarchical Chunking** for structure-aware processing.  
- Watch the progress bar and API quota warnings.

3. **Search**  
- Go to the **Search** tab.  
- Enter a natural-language query.  
- Choose **Hierarchical** or **Standard** mode.  
- View the AI-generated answer and, for hierarchical mode, context hierarchy.

4. **Analyze Structure**  
- Open the **Hierarchical Analysis** tab.  
- Inspect chunk level distribution charts.  
- Explore document sections and subsections interactively.

5. **Monitor Quota**  
- Check remaining Cohere API calls in the header or sidebar.  
- The app will auto-throttle when nearing the limit.

## Customization

- **Chunk Sizes & Overlap**  
Edit `chunk_sizes` and `overlap_ratios` in `hierarchical_chunker.py`.
- **Header Detection**  
Add or adjust regex patterns in the `_extract_headers` method to suit your PDF formats.
- **Rate Limit Settings**  
Tweak `max_calls` and `period` in `rate_limiter.py` for different quotas.
- **LLM Model**  
Change `GEMINI_MODEL` in `config.py` to use alternative Gemini variants.

## Troubleshooting

- **429 Rate Limit Errors**  
Confirm the rate limiter is active and reduce document processing concurrency.
- **KeyError: 'source'**  
Ensure `search.py` uses `doc_info.get("source", "Unknown Source")`.
- **Answer Bolded**  
Verify you use `st.write(answer)` or `st.markdown(f"{answer}")`, not `**{answer}**`.
