# enhanced_app.py (UPDATED VERSION WITH RATE LIMITING DISPLAY)
# Add these modifications to your existing enhanced_app.py

import streamlit as st
from PIL import Image
import uuid
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import enhanced modules with rate limiting
from embedding_rate_limit import get_document_embedding, get_query_embedding, get_rate_limiter_stats
from enhanced_documents_utils import (
    pdf_to_images,
    extract_text_hierarchically,
    save_image_preview,
    load_embeddings_and_info,
    save_hierarchical_embeddings_and_info,
    process_documents_hierarchically,
    get_chunk_context,
    find_best_hierarchical_match
)
from hierarchical_chunker import HierarchicalSearcher
from search import search_documents, answer_with_gemini

# Load session state with hierarchical support
if 'faiss_index' not in st.session_state:
    index, docs_info, relationships = load_embeddings_and_info()
    st.session_state.faiss_index = index
    st.session_state.docs_info = docs_info
    st.session_state.chunk_relationships = relationships

if 'embedding_buffer' not in st.session_state:
    st.session_state.embedding_buffer = []

if 'hierarchical_searcher' not in st.session_state:
    # Initialize hierarchical searcher if we have hierarchical chunks
    hierarchical_chunks = []
    for doc in st.session_state.docs_info:
        if doc.get('content_type') == 'hierarchical_text':
            # Reconstruct hierarchical chunks from docs_info
            from hierarchical_chunker import HierarchicalChunk
            chunk = HierarchicalChunk(
                id=doc['doc_id'],
                content=doc['content'],
                level=doc.get('level', 0),
                parent_id=doc.get('parent_id'),
                children_ids=doc.get('children_ids', []),
                metadata=doc.get('metadata', {}),
                start_page=doc.get('start_page')
            )
            hierarchical_chunks.append(chunk)
    
    if hierarchical_chunks:
        st.session_state.hierarchical_searcher = HierarchicalSearcher(hierarchical_chunks)
    else:
        st.session_state.hierarchical_searcher = None

st.set_page_config(page_title="Enhanced Multimodal RAG", layout="wide")
st.title("🚀 Enhanced Multimodal RAG with Hierarchical Chunking")

# ADD THIS: App status and rate limiting info
col1, col2 = st.columns([3, 1])
with col1:
    st.success("✅ App is running! All systems operational.")
with col2:
    # Display rate limiter stats
    try:
        stats = get_rate_limiter_stats()
        remaining = stats['remaining_calls']
        total = stats['max_calls']
        
        if remaining > 20:
            st.info(f"🚦 API Calls: {remaining}/{total} remaining")
        elif remaining > 10:
            st.warning(f"🚦 API Calls: {remaining}/{total} remaining")
        else:
            st.error(f"🚦 API Calls: {remaining}/{total} remaining")
            
    except Exception:
        st.info("🚦 Rate limiter active")

# Add information about hierarchical chunking and rate limiting
with st.expander("ℹ️ About This Enhanced System"):
    st.markdown("""
    **Hierarchical Chunking** organizes your documents into structured levels:
    - **Level 0**: Entire document
    - **Level 1**: Major sections/chapters  
    - **Level 2**: Subsections
    - **Level 3**: Paragraphs
    - **Level 4**: Sentences
    
    **Rate Limiting**: This app uses Cohere Trial API with 100 calls/minute limit.
    Built-in rate limiting prevents API errors and shows remaining quota.
    """)

tab1, tab2, tab3 = st.tabs(["📚 Index Documents", "🔍 Search", "📊 Hierarchical Analysis"])

# ------------------- Tab 1: Enhanced Indexing ------------------- #
with tab1:
    st.header("Index Your Documents with Hierarchical Chunking")
    
    # Display rate limit warning for document processing
    try:
        stats = get_rate_limiter_stats()
        if stats['remaining_calls'] < 10:
            st.warning("⚠️ Low API quota remaining. Document processing may be slow due to rate limiting.")
    except Exception:
        pass
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDF Reports", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload PDFs to create hierarchical chunks that preserve document structure"
        )
    
    with col2:
        use_hierarchical = st.checkbox(
            "Use Hierarchical Chunking", 
            value=True,
            help="Enable structure-aware chunking for better context preservation"
        )
    
    if uploaded_files and st.button("🔄 Process Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)
        
        if use_hierarchical:
            # Use enhanced hierarchical processing with rate limiting
            status_text.text("Using hierarchical chunking with rate limiting for API calls...")
            
            try:
                new_embeddings, hierarchical_chunks, enhanced_docs_info = process_documents_hierarchically(
                    uploaded_files, get_document_embedding
                )
                
                # Save with hierarchical relationships
                if new_embeddings:
                    save_hierarchical_embeddings_and_info(
                        new_embeddings, 
                        hierarchical_chunks, 
                        enhanced_docs_info
                    )
                    
                    # Update session state
                    index, docs_info, relationships = load_embeddings_and_info()
                    st.session_state.faiss_index = index
                    st.session_state.docs_info = docs_info
                    st.session_state.chunk_relationships = relationships
                    
                    # Update hierarchical searcher
                    if hierarchical_chunks:
                        st.session_state.hierarchical_searcher = HierarchicalSearcher(hierarchical_chunks)
                    
                    st.success(f"✅ Processed {total_files} documents with hierarchical chunking!")
                    
                    # Show chunking statistics
                    level_counts = {}
                    for chunk in hierarchical_chunks:
                        level = chunk.level
                        level_counts[level] = level_counts.get(level, 0) + 1
                    
                    st.info(f"Created {len(hierarchical_chunks)} hierarchical chunks across {len(level_counts)} levels")
                    
                    # Display level distribution
                    if level_counts:
                        df_levels = pd.DataFrame(
                            list(level_counts.items()), 
                            columns=['Level', 'Count']
                        )
                        st.bar_chart(df_levels.set_index('Level'))
                        
                else:
                    st.warning("No embeddings were created. Check your documents and API quota.")
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")
                if "429" in str(e) or "rate limit" in str(e).lower():
                    st.info("💡 This appears to be a rate limiting issue. The system will automatically wait and retry.")
            
        else:
            # Use original processing method
            status_text.text("Using standard chunking...")
            st.warning("Standard chunking selected. For better results, enable hierarchical chunking.")

# ------------------- Tab 2: Enhanced Search ------------------- #
with tab2:
    st.header("🤖 Intelligent Search with Hierarchical Context")
    
    # Show rate limit status for search
    try:
        stats = get_rate_limiter_stats()
        if stats['remaining_calls'] < 5:
            st.warning("⚠️ Very low API quota. Search may trigger rate limiting delays.")
    except Exception:
        pass
    
    # Search configuration
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query", 
            placeholder="e.g., What are the main findings about revenue growth?",
            help="Ask questions about your documents. Hierarchical search will find the most relevant context."
        )
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            ["Hierarchical", "Standard"],
            help="Hierarchical mode uses document structure for better results"
        )
    
    if query:
        if st.session_state.faiss_index is None:
            st.warning("⚠️ No documents indexed yet. Please upload and process documents first.")
        else:
            # Show that rate limiting is active during search
            with st.spinner("🔍 Searching with rate-limited API calls..."):
                # Perform search
                results = search_documents(
                    query, 
                    st.session_state.faiss_index, 
                    st.session_state.docs_info, 
                    get_query_embedding, 
                    top_k=5
                )
            
            if not results:
                st.warning("🔍 No relevant results found. Try rephrasing your query.")
            else:
                if search_mode == "Hierarchical" and st.session_state.hierarchical_searcher:
                    # Use hierarchical search enhancement
                    enhanced_results = st.session_state.hierarchical_searcher.search_with_context(results)
                    best_match = find_best_hierarchical_match(
                        query, 
                        enhanced_results, 
                        st.session_state.docs_info, 
                        st.session_state.chunk_relationships
                    )
                    
                    if best_match:
                        with st.spinner("🧠 Generating enhanced answer (rate-limited API call)..."):
                            # Use the best match for answer generation
                            content = best_match.get('content', '')
                            if best_match.get('content_type') == 'image':
                                content = Image.open(best_match['preview'])
                            
                            answer = answer_with_gemini(query, content)
                            
                            # Display enhanced answer with context
                            st.markdown("### 🎯 Enhanced Answer")
                            st.markdown(f"**{answer}**")
                            
                            # Show hierarchical context
                            hierarchical_context = best_match.get('hierarchical_context', {})
                            if hierarchical_context:
                                with st.expander("📋 Hierarchical Context"):
                                    level = hierarchical_context.get('level', 0)
                                    st.write(f"**Chunk Level:** {level}")
                                    
                                    parent = hierarchical_context.get('parent')
                                    if parent:
                                        st.write(f"**Parent Section:** {parent.get('title', 'Unknown')}")
                                        st.write(f"**Parent Preview:** {parent.get('content_preview', '')[:150]}...")
                                    
                                    children = hierarchical_context.get('children', [])
                                    if children:
                                        st.write(f"**Subsections:** {len(children)} found")
                                        for child in children[:3]:  # Show first 3
                                            st.write(f"- {child.get('title', 'Subsection')}")
                            
                            # Show source information
                            st.markdown("### 📄 Source Information")
                            source_info = best_match.get('source', 'Unknown')
                            page_info = best_match.get('start_page', 'Unknown')
                            st.info(f"**Source:** {source_info} | **Page:** {page_info}")
                            
                else:
                    # Standard search mode
                    text_result = next((r for r in results if r['content_type'] == 'text' or r['content_type'] == 'hierarchical_text'), None)
                    image_result = next((r for r in results if r['content_type'] == 'image'), None)
                    
                    with st.spinner("🤖 Generating answer (rate-limited)..."):
                        if image_result:
                            content = Image.open(image_result['preview'])
                        elif text_result:
                            content = text_result['content']
                        else:
                            content = ""
                        
                        answer = answer_with_gemini(query, content)
                        st.markdown(f"### 🤖 Answer:\n{answer}")

                
                # Display search results
                st.markdown("### 🔍 Search Results")
                for i, result in enumerate(results[:3]):
                    with st.expander(f"Result {i+1} - {result.get('source', 'Unknown')} (Score: {result.get('similarity', 0):.3f})"):
                        if result.get('content_type') == 'image':
                            st.image(Image.open(result['preview']), width=400)
                        else:
                            st.write(result.get('content', result.get('preview', ''))[:500] + "...")

# ------------------- Tab 3: Hierarchical Analysis ------------------- #
with tab3:
    st.header("📊 Hierarchical Document Analysis")
    
    if not st.session_state.docs_info:
        st.info("📚 No documents indexed yet. Upload documents to see hierarchical analysis.")
    else:
        # Filter hierarchical chunks
        hierarchical_docs = [doc for doc in st.session_state.docs_info if doc.get('content_type') == 'hierarchical_text']
        
        if not hierarchical_docs:
            st.warning("⚠️ No hierarchical chunks found. Enable hierarchical chunking when processing documents.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Chunk Level Distribution")
                level_counts = {}
                for doc in hierarchical_docs:
                    level = doc.get('level', 0)
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                if level_counts:
                    fig, ax = plt.subplots()
                    levels = list(level_counts.keys())
                    counts = list(level_counts.values())
                    ax.bar([f"Level {l}" for l in levels], counts)
                    ax.set_title("Distribution of Chunks by Hierarchy Level")
                    ax.set_xlabel("Hierarchy Level")
                    ax.set_ylabel("Number of Chunks")
                    st.pyplot(fig)
            
            with col2:
                st.subheader("🏗️ Document Structure")
                # Show document structure
                sources = {}
                for doc in hierarchical_docs:
                    source = doc.get('source', 'Unknown')
                    level = doc.get('level', 0)
                    if source not in sources:
                        sources[source] = {}
                    sources[source][level] = sources[source].get(level, 0) + 1
                
                for source, levels in sources.items():
                    st.write(f"**📄 {source}**")
                    for level, count in sorted(levels.items()):
                        st.write(f"  - Level {level}: {count} chunks")
            
            st.subheader("🔍 Explore Hierarchical Structure")
            
            # Document selector
            available_sources = list(set(doc.get('source', 'Unknown') for doc in hierarchical_docs))
            selected_source = st.selectbox("Select Document", available_sources)
            
            if selected_source:
                # Show hierarchical structure for selected document
                source_docs = [doc for doc in hierarchical_docs if doc.get('source') == selected_source]
                
                # Group by level
                levels = {}
                for doc in source_docs:
                    level = doc.get('level', 0)
                    if level not in levels:
                        levels[level] = []
                    levels[level].append(doc)
                
                for level in sorted(levels.keys()):
                    with st.expander(f"Level {level} ({len(levels[level])} chunks)"):
                        for doc in levels[level][:5]:  # Show first 5 chunks
                            title = doc.get('metadata', {}).get('title', f"Chunk {doc.get('doc_id', '')[:8]}")
                            preview = doc.get('preview', '')[:200]
                            st.write(f"**{title}**")
                            st.write(f"{preview}...")
                            st.write("---")

# ------------------- Enhanced Sidebar with Rate Limiting Stats ------------------- #
with st.sidebar:
    st.header("📊 Enhanced Index Stats")
    
    # Rate limiting status
    st.subheader("🚦 API Rate Limiting")
    try:
        stats = get_rate_limiter_stats()
        remaining = stats['remaining_calls']
        total = stats['max_calls']
        used_percentage = ((total - remaining) / total) * 100
        
        st.metric("Remaining API Calls", f"{remaining}/{total}")
        st.progress(used_percentage / 100)
        
        if remaining < 10:
            st.warning("⚠️ Low quota - operations will be slower")
        elif remaining < 30:
            st.info("ℹ️ Moderate quota remaining")
        else:
            st.success("✅ Good quota available")
            
    except Exception as e:
        st.error(f"Rate limiter error: {e}")
    
    if st.session_state.faiss_index is not None and st.session_state.docs_info:
        total_items = len(st.session_state.docs_info)
        st.metric("Total Indexed Items", total_items)
        
        # Content type distribution
        content_types = [doc.get("content_type", "unknown") for doc in st.session_state.docs_info]
        type_counts = pd.Series(content_types).value_counts()
        
        st.subheader("Content Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax.set_title("Content Type Distribution")
        st.pyplot(fig)
        
        # Hierarchical stats
        hierarchical_docs = [doc for doc in st.session_state.docs_info if doc.get('content_type') == 'hierarchical_text']
        if hierarchical_docs:
            st.subheader("Hierarchical Stats")
            levels = [doc.get('level', 0) for doc in hierarchical_docs]
            avg_level = sum(levels) / len(levels) if levels else 0
            max_level = max(levels) if levels else 0
            
            st.metric("Hierarchical Chunks", len(hierarchical_docs))
            st.metric("Average Level", f"{avg_level:.1f}")
            st.metric("Max Depth", max_level)
        
        # Clear data option
        st.subheader("🗑️ Data Management")
        if st.button("Clear All Data", type="secondary"):
            try:
                # Clear all data files
                for file_name in ["faiss.index", "docs_info.pkl", "chunk_relationships.pkl"]:
                    file_path = os.path.join("data", file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                # Reset session state
                st.session_state.faiss_index = None
                st.session_state.docs_info = []
                st.session_state.chunk_relationships = {}
                st.session_state.hierarchical_searcher = None
                
                st.success("✅ All data cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing data: {e}")
    else:
        st.write("📝 No documents indexed yet.")
        st.write("Upload documents to see statistics.")

# Footer
st.markdown("---")
st.markdown("🚀 **Enhanced Multimodal RAG** - Powered by Hierarchical Chunking & Rate Limiting")