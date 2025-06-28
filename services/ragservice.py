"""
RAG Service - Retrieval Augmented Generation
Integrates document processing, vector search, and LLM generation
"""

import os
from typing import List, Dict, Optional
from .documentprocessor import DocumentProcessor
from .simplevectorstore import VectorStoreService
from vqa import *
import requests
import json

class RAGService:
    def __init__(self, 
                 documents_dir: str = "documents",
                 vector_store_path: str = "medical_vector_store.json",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize RAG service
        
        Args:
            documents_dir: Directory containing medical documents
            vector_store_path: Path to vector store file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStoreService(vector_store_path)
        self.api_key = self._get_api_key()
        
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config file"""
        try:
            with open('config/openai_api_key', 'r') as file:
                api_key = file.read().strip()
            if not api_key:
                raise ValueError("API key not found in config file")
            return api_key
        except Exception as e:
            print(f"Error loading API key: {str(e)}")
            return None
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the medical knowledge base from documents
        
        Args:
            force_rebuild: Whether to rebuild even if vector store exists
            
        Returns:
            True if successful, False otherwise
        """
        # Check if vector store already exists and has content
        store_info = self.vector_store.get_store_info()
        if store_info['total_vectors'] > 0 and not force_rebuild:
            print(f"Knowledge base already initialized with {store_info['total_vectors']} vectors")
            print(f"Sources: {store_info['sources']}")
            return True
        
        print("Initializing medical knowledge base...")
        
        # Check if documents directory exists
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory {self.documents_dir} not found!")
            return False
        
        try:
            # Process all PDF documents
            print(f"Processing documents from {self.documents_dir}...")
            all_chunks = self.doc_processor.process_documents_directory(self.documents_dir)
            
            if not all_chunks:
                print("No documents were processed successfully!")
                return False
            
            # Filter for medical content
            print("Filtering for medical content...")
            medical_chunks = self.doc_processor.filter_medical_content(all_chunks)
            
            print(f"Found {len(medical_chunks)} relevant medical text chunks")
            
            if not medical_chunks:
                print("No relevant medical content found!")
                return False
            
            # Add to vector store
            print("Adding documents to vector store...")
            self.vector_store.add_documents(medical_chunks)
            
            # Print summary
            store_info = self.vector_store.get_store_info()
            print(f"\n✅ Knowledge base initialized successfully!")
            print(f"Total vectors: {store_info['total_vectors']}")
            print(f"Sources: {store_info['sources']}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
            return False
    
    def get_relevant_medical_context(self, query: str, max_context_length: int = 3000) -> str:
        """
        Retrieve relevant medical context for a query
        
        Args:
            query: User question or analysis context
            max_context_length: Maximum length of context
            
        Returns:
            Relevant medical literature context
        """
        try:
            context = self.vector_store.get_relevant_context(query, max_context_length)
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def analyze_retinal_image_and_heatmap(self, original_image, heatmap_figure, prediction_results, patient_age=None, diabetes_duration=None):
        """Analyze retinal image with heatmap using GPT-4o-mini vision capabilities enhanced with RAG"""
        api_url = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("API key not found")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Encode images to base64
        original_b64 = encode_image_to_base64(original_image)
        heatmap_b64 = encode_image_to_base64(heatmap_figure)

        # Construct detailed prompt
        patient_info = ""
        if patient_age and diabetes_duration:
            patient_info = f"\nPatient Information:\n- Age: {patient_age} years\n- Duration of diabetes: {diabetes_duration} years\n"

        # Get relevant medical literature context using RAG
        medical_context = ""
        try:
            search_query = f"diabetic retinopathy grade {prediction_results['value']} {prediction_results['class']} analysis heatmap fundus examination"
            medical_context = self.get_relevant_medical_context(search_query, max_context_length=2000)
        except Exception as e:
            print(f"Warning: Could not retrieve medical context: {e}")

        # Enhanced prompt with medical literature context
        base_prompt = f"""You are an expert ophthalmologist AI assistant analyzing retinal images for diabetic retinopathy.

        {patient_info}
        AI Model Results:
        - Predicted Class: {prediction_results['class']}
        - Severity Grade: {prediction_results['value']}
        - Confidence: {prediction_results['probability']:.2%}

        I'm showing you two images:
        1. The original retinal fundus photograph
        2. A GradCAM heatmap visualization showing which areas the AI model focused on for its prediction"""

        if medical_context:
            enhanced_prompt = f"""{base_prompt}

        RELEVANT MEDICAL LITERATURE:
        {medical_context}

        Please provide a comprehensive analysis that incorporates insights from the current medical literature above, including:

        1. **Clinical Assessment**: Explain what the AI prediction means in medical terms, referencing relevant literature
        2. **Heatmap Analysis**: Describe what the highlighted areas represent and their clinical significance based on current research
        3. **Key Findings**: Identify specific retinal features visible in the image that support the diagnosis, citing literature when relevant
        4. **Evidence-Based Patient Explanation**: Provide a clear, patient-friendly explanation supported by research findings
        5. **Current Guidelines Recommendations**: Suggest appropriate next steps based on the latest clinical guidelines
        6. **Monitoring Protocol**: Advise on follow-up frequency and warning signs based on evidence-based practices

        **IMPORTANT FORMATTING INSTRUCTIONS:**
        - When referencing literature or research findings, use the format: ***According to the literature, [finding]*** or ***Research indicates that [finding]*** or ***Studies show that [finding]***
        - Make all literature citations bold and italic using ***text*** format
        - This will help patients easily identify evidence-based information
        - Example: ***According to recent studies, GradCAM highlighted areas typically indicate microaneurysms which are early signs of diabetic retinopathy***

        When relevant, cite the medical literature to support your analysis and recommendations."""
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert ophthalmologist specializing in diabetic retinopathy analysis. You have access to current medical literature and provide detailed, accurate medical insights while being accessible to patients. Always use bold italic formatting when citing literature or research findings to make evidence-based information clearly visible."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{original_b64}"}
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{heatmap_b64}"}
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    def get_knowledge_base_stats(self) -> Dict[str, any]:
        """Get statistics about the knowledge base"""
        return self.vector_store.get_store_info()
    
    def search_medical_literature(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Search medical literature directly
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        return self.vector_store.search_similar_documents(query, k)
    
    def rebuild_knowledge_base(self) -> bool:
        """Rebuild the knowledge base from scratch"""
        print("Rebuilding knowledge base...")
        self.vector_store.clear_store()
        return self.initialize_knowledge_base(force_rebuild=True) 