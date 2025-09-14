# enhanced_document_utils.py
# Modified version of your document_utils.py with hierarchical chunking

import os
import io
import tempfile
from PIL import Image
import pdf2image
import PyPDF2
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from hierarchical_chunker import HierarchicalChunker, HierarchicalChunk, HierarchicalSearcher

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def pdf_to_images(pdf_file):
    """Convert PDF to images (unchanged from original)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    images = pdf2image.convert_from_path(tmp_path, dpi=200)
    os.unlink(tmp_path)
    return images

def extract_text_hierarchically(pdf_file):
    """Enhanced text extraction with hierarchical structure"""
    try:
        # Initialize hierarchical chunker
        chunker = HierarchicalChunker()
        
        # Extract structured content
        structure = chunker.extract_pdf_structure(pdf_file)
        
        # Create hierarchical chunks
        hierarchical_chunks = chunker.chunk_hierarchically(structure)
        
        return hierarchical_chunks, structure
        
    except Exception as e:
        print(f"Hierarchical text extraction error: {e}")
        # Fallback to original method
        return extract_text_from_pdf_fallback(pdf_file), {}

def extract_text_from_pdf_fallback(pdf_file):
    """Original text extraction as fallback"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def save_image_preview(image, filename):
    """Save image preview (unchanged from original)"""
    path = os.path.join(DATA_DIR, filename)
    image.save(path)
    return path

def save_hierarchical_embeddings_and_info(embeddings_data, hierarchical_chunks, docs_info):
    """Save embeddings with hierarchical relationship information"""
    
    # Prepare vectors for FAISS
    vectors = [item["embedding"].astype("float32") for item in embeddings_data]
    
    if vectors:
        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(np.vstack(vectors))
        faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))
    
    # Enhanced docs_info with hierarchical metadata
    enhanced_docs_info = []
    
    # Process hierarchical chunks
    chunk_map = {chunk.id: chunk for chunk in hierarchical_chunks}
    
    for i, chunk in enumerate(hierarchical_chunks):
        if i < len(embeddings_data):  # Ensure we have corresponding embedding
            doc_info = {
                "doc_id": chunk.id,
                "content": chunk.content,
                "content_type": "hierarchical_text",
                "level": chunk.level,
                "parent_id": chunk.parent_id,
                "children_ids": chunk.children_ids,
                "metadata": chunk.metadata,
                "start_page": chunk.start_page,
                "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            }
            enhanced_docs_info.append(doc_info)
    
    # Add regular docs_info (for images, etc.)
    enhanced_docs_info.extend(docs_info)
    
    # Save enhanced docs info
    with open(os.path.join(DATA_DIR, "docs_info.pkl"), "wb") as f:
        pickle.dump(enhanced_docs_info, f)
    
    # Save hierarchical relationships separately for easy access
    chunk_relationships = {
        chunk.id: {
            'parent_id': chunk.parent_id,
            'children_ids': chunk.children_ids,
            'level': chunk.level,
            'metadata': chunk.metadata
        } for chunk in hierarchical_chunks
    }
    
    with open(os.path.join(DATA_DIR, "chunk_relationships.pkl"), "wb") as f:
        pickle.dump(chunk_relationships, f)

def save_embeddings_and_info(embeddings_data, docs_info):
    """Original save function for backward compatibility"""
    if not embeddings_data:
        return
        
    vectors = [item["embedding"].astype("float32") for item in embeddings_data]
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.vstack(vectors))

    faiss.write_index(index, os.path.join(DATA_DIR, "faiss.index"))

    with open(os.path.join(DATA_DIR, "docs_info.pkl"), "wb") as f:
        pickle.dump(docs_info, f)

def load_embeddings_and_info():
    """Load embeddings with hierarchical support"""
    index_path = os.path.join(DATA_DIR, "faiss.index")
    docs_path = os.path.join(DATA_DIR, "docs_info.pkl")
    relationships_path = os.path.join(DATA_DIR, "chunk_relationships.pkl")

    if os.path.exists(index_path) and os.path.exists(docs_path):
        index = faiss.read_index(index_path)
        
        with open(docs_path, "rb") as f:
            docs_info = pickle.load(f)
        
        # Load relationships if available
        relationships = {}
        if os.path.exists(relationships_path):
            with open(relationships_path, "rb") as f:
                relationships = pickle.load(f)
        
        return index, docs_info, relationships
    else:
        return None, [], {}

# Enhanced processing function for your app.py
def process_documents_hierarchically(uploaded_files, get_document_embedding_func):
    """
    Process uploaded files with hierarchical chunking
    
    Args:
        uploaded_files: List of uploaded PDF files
        get_document_embedding_func: Function to get embeddings
        
    Returns:
        Tuple of (new_embeddings, hierarchical_chunks, enhanced_docs_info)
    """
    new_embeddings = []
    all_hierarchical_chunks = []
    enhanced_docs_info = []
    
    for uploaded_file in uploaded_files:
        try:
            # Extract hierarchical chunks
            hierarchical_chunks, structure = extract_text_hierarchically(uploaded_file)
            
            if not hierarchical_chunks:
                continue
            
            # Process text chunks
            for chunk in hierarchical_chunks:
                if chunk.content.strip():
                    emb = get_document_embedding_func(chunk.content, "text")
                    if emb is not None:
                        new_embeddings.append({
                            "embedding": emb, 
                            "doc_id": chunk.id, 
                            "content_type": "hierarchical_text"
                        })
                        
                        # Create enhanced doc info
                        doc_info = {
                            "doc_id": chunk.id,
                            "source": uploaded_file.name,
                            "content_type": "hierarchical_text",
                            "content": chunk.content,
                            "level": chunk.level,
                            "parent_id": chunk.parent_id,
                            "children_ids": chunk.children_ids,
                            "metadata": chunk.metadata,
                            "start_page": chunk.start_page,
                            "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        }
                        enhanced_docs_info.append(doc_info)
            
            all_hierarchical_chunks.extend(hierarchical_chunks)
            
            # Also process images as before
            images = pdf_to_images(uploaded_file)
            for page_num, img in enumerate(images, 1):
                page_id = f"{uploaded_file.name}_page_{page_num}"
                emb = get_document_embedding_func(img, "image")
                if emb is not None:
                    new_embeddings.append({
                        "embedding": emb, 
                        "doc_id": page_id, 
                        "content_type": "image"
                    })
                    
                    path = save_image_preview(img, f"{page_id}.png")
                    enhanced_docs_info.append({
                        "doc_id": page_id,
                        "source": uploaded_file.name,
                        "content_type": "image",
                        "page": page_num,
                        "preview": path,
                    })
                    
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {e}")
            continue
    
    return new_embeddings, all_hierarchical_chunks, enhanced_docs_info

# Utility functions for hierarchical search
def get_chunk_context(doc_id: str, docs_info: List[Dict], relationships: Dict) -> Dict[str, Any]:
    """Get hierarchical context for a chunk"""
    
    # Find the chunk info
    chunk_info = next((doc for doc in docs_info if doc['doc_id'] == doc_id), None)
    if not chunk_info:
        return {}
    
    context = {
        'current': chunk_info,
        'level': chunk_info.get('level', 0),
        'parent': None,
        'children': [],
        'siblings': []
    }
    
    # Get parent context
    parent_id = chunk_info.get('parent_id')
    if parent_id:
        parent_info = next((doc for doc in docs_info if doc['doc_id'] == parent_id), None)
        if parent_info:
            context['parent'] = {
                'title': parent_info.get('metadata', {}).get('title', ''),
                'content_preview': parent_info.get('preview', ''),
                'level': parent_info.get('level', 0)
            }
    
    # Get children context
    children_ids = chunk_info.get('children_ids', [])
    for child_id in children_ids:
        child_info = next((doc for doc in docs_info if doc['doc_id'] == child_id), None)
        if child_info:
            context['children'].append({
                'title': child_info.get('metadata', {}).get('title', ''),
                'content_preview': child_info.get('preview', ''),
                'level': child_info.get('level', 0)
            })
    
    return context

def find_best_hierarchical_match(query: str, search_results: List[Dict], docs_info: List[Dict], relationships: Dict) -> Dict:
    """Find the best hierarchical match considering context"""
    
    if not search_results:
        return {}
    
    # Score results based on hierarchical relevance
    scored_results = []
    
    for result in search_results:
        score = result.get('similarity', 0)
        doc_id = result.get('doc_id', '')
        
        # Get hierarchical context
        context = get_chunk_context(doc_id, docs_info, relationships)
        
        # Boost score based on hierarchical features
        level = context.get('level', 0)
        
        # Prefer mid-level chunks (not too high, not too low)
        if 1 <= level <= 3:
            score += 0.1
        
        # Boost if has meaningful parent context
        if context.get('parent'):
            score += 0.05
        
        # Boost if has children (comprehensive sections)
        if context.get('children'):
            score += 0.05
        
        scored_results.append({
            **result,
            'adjusted_score': score,
            'hierarchical_context': context
        })
    
    # Return best scored result
    return max(scored_results, key=lambda x: x.get('adjusted_score', 0))