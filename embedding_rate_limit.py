# embeddings.py (REPLACE YOUR EXISTING FILE WITH THIS)
# Rate-limited version of your embeddings.py

import numpy as np
import io
import base64
from PIL import Image
from config import COHERE_API_KEY
import cohere
from rate_limiter import cohere_rate_limiter

# Constants
MAX_PIXELS = 1568 * 1568  # Cohere image size limit

# Initialize Cohere client
co_client = cohere.ClientV2(api_key=COHERE_API_KEY)

def resize_image(pil_image):
    """Resize image if too large for embedding API"""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        return pil_image.resize((new_width, new_height))
    return pil_image

def base64_from_image(pil_image):
    """Convert PIL Image to base64 for Cohere"""
    pil_image = resize_image(pil_image)
    img_format = pil_image.format if pil_image.format else "PNG"
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
        return f"data:image/{img_format.lower()};base64," + base64.b64encode(img_bytes).decode("utf-8")

def get_document_embedding(content, content_type="text"):
    """
    Embed document (text or image) with rate limiting
    """
    # Apply rate limiting before making API call
    cohere_rate_limiter.acquire()
    
    try:
        if content_type == "text":
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=[content],
            )
            return np.array(response.embeddings.float[0])
        else:
            api_input_document = {
                "content": [
                    {"type": "image", "image": base64_from_image(content)},
                ]
            }
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                inputs=[api_input_document],
            )
            return np.array(response.embeddings.float[0])
    
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def get_query_embedding(query):
    """
    Embed search query with rate limiting
    """
    # Apply rate limiting before making API call
    cohere_rate_limiter.acquire()
    
    try:
        response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[query],
        )
        return np.array(response.embeddings.float[0])
    
    except Exception as e:
        print(f"Query embedding error: {e}")
        return None

# Batch embedding function for better efficiency
def get_batch_embeddings(texts, content_type="text", batch_size=10):
    """
    Get embeddings for multiple texts in batches with rate limiting
    
    Args:
        texts: List of text strings to embed
        content_type: Type of content ("text" only for batch)
        batch_size: Number of texts to process in each batch
    
    Returns:
        List of numpy arrays (embeddings)
    """
    if content_type != "text":
        raise ValueError("Batch embeddings only supported for text content")
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Apply rate limiting before batch API call
        cohere_rate_limiter.acquire()
        
        try:
            response = co_client.embed(
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
                texts=batch,
            )
            
            batch_embeddings = [np.array(emb) for emb in response.embeddings.float]
            embeddings.extend(batch_embeddings)
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Batch embedding error: {e}")
            # Add None for failed embeddings
            embeddings.extend([None] * len(batch))
    
    return embeddings

def get_rate_limiter_stats():
    """Get current rate limiter statistics"""
    return cohere_rate_limiter.get_stats()