
"""
Document processing service for RAG system
Handles PDF extraction, text chunking, and preprocessing
"""

import os
import re
import fitz
from typing import List, Dict, Tuple
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor

        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()

            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
    # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\%\+\=\<\>]', '', text)

    # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

