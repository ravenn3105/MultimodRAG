# hierarchical_chunker.py
# Advanced hierarchical chunking implementation for multimodal RAG

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import PyPDF2
import io
import os
import pickle

@dataclass
class HierarchicalChunk:
    """Represents a chunk in the hierarchical structure"""
    id: str
    content: str
    level: int
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
    start_page: Optional[int] = None
    end_page: Optional[int] = None

class HierarchicalChunker:
    """
    Advanced hierarchical chunking implementation for your RAG project
    """
    
    def __init__(self, 
                 chunk_sizes: List[int] = [2048, 1024, 512, 256],
                 overlap_ratios: List[float] = [0.1, 0.15, 0.2, 0.25]):
        """
        Initialize hierarchical chunker
        
        Args:
            chunk_sizes: Token sizes for each hierarchy level
            overlap_ratios: Overlap ratios for each level
        """
        self.chunk_sizes = chunk_sizes
        self.overlap_ratios = overlap_ratios
        self.separator_hierarchy = [
            r'\n\n\n+',  # Multiple newlines (section breaks)
            r'\n\n',      # Double newlines (paragraph breaks)
            r'\n',         # Single newlines
            r'\. ',        # Sentence endings
            r' ',           # Word boundaries
            r''             # Character level
        ]
    
    def extract_pdf_structure(self, pdf_file) -> Dict[str, Any]:
        """
        Extract hierarchical structure from PDF
        
        Args:
            pdf_file: PDF file object
            
        Returns:
            Dictionary containing structured content
        """
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
            
            # Extract text with page information
            pages_content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages_content.append({
                        'page_num': page_num + 1,
                        'content': text,
                        'headers': self._extract_headers(text)
                    })
            
            # Build document structure
            structure = self._build_document_structure(pages_content)
            return structure
            
        except Exception as e:
            print(f"Error extracting PDF structure: {e}")
            return {'pages': [], 'structure': {}}
    
    def _extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential headers from text based on patterns"""
        headers = []
        
        # Common header patterns
        patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^\d+\.\s+(.+)$',   # Numbered sections (1. Title)
            r'^\d+\.\d+\s+(.+)$',  # Subsections (1.1 Title)
            r'^[A-Z][A-Z\s]+$',    # ALL CAPS headers
            r'^[A-Z].{1,50}[^.!?]$'  # Title case headers
        ]
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines):
            line = line.strip()
            if len(line) > 3 and len(line) < 100:  # Reasonable header length
                for pattern in patterns:
                    if re.match(pattern, line, re.MULTILINE):
                        level = self._determine_header_level(line)
                        headers.append({
                            'text': line,
                            'level': level,
                            'line_num': line_num,
                            'pattern': pattern
                        })
                        break
        
        return headers
    
    def _determine_header_level(self, header_text: str) -> int:
        """Determine hierarchy level based on header characteristics"""
        # Markdown headers
        if header_text.startswith('#'):
            return header_text.count('#')
        
        # Numbered sections
        if re.match(r'^\d+\.\s+', header_text):
            return 1
        elif re.match(r'^\d+\.\d+\s+', header_text):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+\s+', header_text):
            return 3
        
        # ALL CAPS (usually major sections)
        if header_text.isupper():
            return 1
        
        # Default
        return 2
    
    def _build_document_structure(self, pages_content: List[Dict]) -> Dict[str, Any]:
        """Build hierarchical document structure"""
        structure = {
            'sections': [],
            'metadata': {
                'total_pages': len(pages_content),
                'total_headers': 0
            }
        }
        
        current_section = None
        
        for page in pages_content:
            headers = page['headers']
            structure['metadata']['total_headers'] += len(headers)
            
            if headers:
                # Process headers to create sections
                for header in headers:
                    section = {
                        'title': header['text'],
                        'level': header['level'],
                        'start_page': page['page_num'],
                        'content': [],
                        'subsections': []
                    }
                    
                    if header['level'] == 1:
                        structure['sections'].append(section)
                        current_section = section
                    elif current_section and header['level'] > current_section.get('level', 1):
                        current_section['subsections'].append(section)
            
            # Add page content
            if current_section:
                current_section['content'].append(page['content'])
            else:
                # Create default section if no headers found
                if not structure['sections']:
                    structure['sections'].append({
                        'title': 'Document Content',
                        'level': 1,
                        'start_page': 1,
                        'content': [],
                        'subsections': []
                    })
                structure['sections'][0]['content'].append(page['content'])
        
        return structure
    
    def chunk_hierarchically(self, document_structure: Dict[str, Any]) -> List[HierarchicalChunk]:
        """
        Create hierarchical chunks from document structure
        
        Args:
            document_structure: Structured document content
            
        Returns:
            List of hierarchical chunks
        """
        chunks = []
        
        # Level 0: Document level
        full_content = self._extract_full_content(document_structure)
        doc_chunk = HierarchicalChunk(
            id=str(uuid.uuid4()),
            content=full_content,
            level=0,
            parent_id=None,
            children_ids=[],
            metadata={'type': 'document', 'chunk_method': 'hierarchical'}
        )
        chunks.append(doc_chunk)
        
        # Process sections hierarchically
        for section in document_structure.get('sections', []):
            section_chunks = self._process_section(section, doc_chunk.id, 1)
            chunks.extend(section_chunks)
            doc_chunk.children_ids.extend([chunk.id for chunk in section_chunks if chunk.level == 1])
        
        return chunks
    
    def _extract_full_content(self, structure: Dict[str, Any]) -> str:
        """Extract all text content from structure"""
        content_parts = []
        
        for section in structure.get('sections', []):
            if section.get('title'):
                content_parts.append(section['title'])
            
            content_parts.extend(section.get('content', []))
            
            for subsection in section.get('subsections', []):
                if subsection.get('title'):
                    content_parts.append(subsection['title'])
                content_parts.extend(subsection.get('content', []))
        
        return '\n'.join(content_parts)
    
    def _process_section(self, section: Dict[str, Any], parent_id: str, level: int) -> List[HierarchicalChunk]:
        """Process a section into hierarchical chunks"""
        chunks = []
        
        # Create section chunk
        section_content = section.get('title', '') + '\n' + '\n'.join(section.get('content', []))
        
        section_chunk = HierarchicalChunk(
            id=str(uuid.uuid4()),
            content=section_content,
            level=level,
            parent_id=parent_id,
            children_ids=[],
            metadata={
                'type': 'section',
                'title': section.get('title', ''),
                'start_page': section.get('start_page')
            }
        )
        chunks.append(section_chunk)
        
        # Process subsections
        for subsection in section.get('subsections', []):
            subsection_chunks = self._process_section(subsection, section_chunk.id, level + 1)
            chunks.extend(subsection_chunks)
            section_chunk.children_ids.extend([chunk.id for chunk in subsection_chunks if chunk.level == level + 1])
        
        # Create smaller chunks if section is too large
        if len(section_content) > self.chunk_sizes[min(level, len(self.chunk_sizes) - 1)]:
            smaller_chunks = self._create_smaller_chunks(
                section_content, 
                section_chunk.id, 
                level + 1,
                section.get('start_page')
            )
            chunks.extend(smaller_chunks)
            section_chunk.children_ids.extend([chunk.id for chunk in smaller_chunks])
        
        return chunks
    
    def _create_smaller_chunks(self, content: str, parent_id: str, level: int, start_page: Optional[int]) -> List[HierarchicalChunk]:
        """Create smaller chunks using recursive splitting"""
        chunks = []
        
        if level >= len(self.chunk_sizes):
            return chunks
        
        chunk_size = self.chunk_sizes[level]
        overlap_ratio = self.overlap_ratios[level]
        overlap_size = int(chunk_size * overlap_ratio)
        
        # Split content
        current_chunks = self._recursive_split(content, chunk_size, overlap_size, level)
        
        for i, chunk_content in enumerate(current_chunks):
            if chunk_content.strip():
                chunk = HierarchicalChunk(
                    id=str(uuid.uuid4()),
                    content=chunk_content,
                    level=level,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={
                        'type': 'chunk',
                        'chunk_index': i,
                        'start_page': start_page
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _recursive_split(self, text: str, chunk_size: int, overlap_size: int, level: int) -> List[str]:
        """Recursively split text using appropriate separators"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        separator_index = min(level, len(self.separator_hierarchy) - 1)
        
        for sep_pattern in self.separator_hierarchy[separator_index:]:
            parts = re.split(sep_pattern, text)
            
            if len(parts) > 1:
                # Successfully split
                current_chunk = ""
                
                for part in parts:
                    if len(current_chunk) + len(part) <= chunk_size:
                        current_chunk += part + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # Fallback: character-based splitting
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_size
        
        return chunks

# Enhanced search functionality for hierarchical chunks
class HierarchicalSearcher:
    """Enhanced search that leverages hierarchical structure"""
    
    def __init__(self, chunks: List[HierarchicalChunk]):
        self.chunks = chunks
        self.chunk_map = {chunk.id: chunk for chunk in chunks}
    
    def search_with_context(self, query_results: List[Dict], expand_context: bool = True) -> List[Dict]:
        """
        Enhance search results with hierarchical context
        
        Args:
            query_results: Initial search results
            expand_context: Whether to include parent context
            
        Returns:
            Enhanced results with hierarchical context
        """
        enhanced_results = []
        
        for result in query_results:
            chunk_id = result.get('doc_id')
            chunk = self.chunk_map.get(chunk_id)
            
            if chunk:
                enhanced_result = result.copy()
                
                # Add hierarchical metadata
                enhanced_result['hierarchical_info'] = {
                    'level': chunk.level,
                    'has_children': len(chunk.children_ids) > 0,
                    'has_parent': chunk.parent_id is not None
                }
                
                if expand_context and chunk.parent_id:
                    parent_chunk = self.chunk_map.get(chunk.parent_id)
                    if parent_chunk:
                        enhanced_result['parent_context'] = {
                            'content': parent_chunk.content[:200] + "...",
                            'title': parent_chunk.metadata.get('title', ''),
                            'level': parent_chunk.level
                        }
                
                enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def get_section_summary(self, chunk_id: str) -> Optional[str]:
        """Get a summary of the section this chunk belongs to"""
        chunk = self.chunk_map.get(chunk_id)
        if not chunk:
            return None
        
        # Find the root section
        current = chunk
        while current.parent_id and current.level > 1:
            current = self.chunk_map.get(current.parent_id)
            if not current:
                break
        
        if current and current.metadata.get('type') == 'section':
            return f"Section: {current.metadata.get('title', 'Unknown')} (Page {current.start_page or 'Unknown'})"
        
        return None