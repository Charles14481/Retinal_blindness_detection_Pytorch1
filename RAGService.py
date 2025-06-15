
"""
RAG Service - Retrieval Augmented Generation
Integrates document processing, vector search, and LLM generation
"""

import os
from typing import List, Dict, Optional
import DocumentProcessor
import SimpleVectorStore
from vqa import *
from requests import *
from json import *


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
        self.vector_store = SimpleVectorStore(vector_store_path)
        self.api_key = self._get_api_key()

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


    def enhanced_medical_analysis(self,
                                original_prompt: str,
                                patient_context: str = "",
                                prediction_results: Dict = None) -> str:
        """
        Provide enhanced medical analysis using RAG

        Args:
            original_prompt: Original analysis prompt
            patient_context: Patient information context
            prediction_results: AI prediction results

        Returns:
            Enhanced analysis with medical literature support
        """
        if not self.api_key:
            return "Error: OpenAI API key not available"

    # Create search query for relevant literature
        search_query = f"diabetic retinopathy {original_prompt}"
        if prediction_results:
            search_query += f" grade {prediction_results.get('value', '')} {prediction_results.get('class', '')}"

    # Get relevant medical context
        medical_context = self.get_relevant_medical_context(search_query, max_context_length=2500)

    # Construct enhanced prompt
        enhanced_prompt = f"""You are an expert ophthalmologist providing analysis based on current medical literature and best practices.

    {patient_context}

    RELEVANT MEDICAL LITERATURE:
    {medical_context}

    ORIGINAL ANALYSIS REQUEST:
    {original_prompt}

    Please provide a comprehensive analysis that:
    1. Incorporates insights from the relevant medical literature above
    2. Follows evidence-based best practices
    3. Cites specific findings from the literature when relevant
    4. Provides both technical medical assessment and patient-friendly explanation
    5. Includes appropriate recommendations based on current guidelines

    **IMPORTANT FORMATTING INSTRUCTIONS:**
    - When referencing literature or research findings, use the format: ***According to the literature, [finding]*** or ***Research indicates that [finding]*** or ***Studies show that [finding]***
    - Make all literature citations bold and italic using ***text*** format
    - This will help patients easily identify evidence-based information
    - Example: ***According to recent studies, patients with moderate diabetic retinopathy have a 25% risk of progression within one year***

    When referencing the literature, mention the source papers to add credibility to your analysis."""

        return self._call_gpt_with_enhanced_prompt(enhanced_prompt)
    
    def analyze_retinal_image_and_heatmap(original_image, heatmap_figure, prediction_results, patient_age=None, diabetes_duration=None):
        """Analyze retinal image with heatmap using GPT-4o-mini vision capabilities enhanced with RAG"""
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = get_api_key()

        if not api_key:
            raise ValueError("API key not found")

        # Initialize RAG service
        rag = initialize_rag()

        headers = {
            "Authorization": f"Bearer {api_key}",
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
        if rag:
            try:
                search_query = f"diabetic retinopathy grade {prediction_results['value']} {prediction_results['class']} analysis heatmap fundus examination"
                medical_context = rag.get_relevant_medical_context(search_query, max_context_length=2000)
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

    def answer_retinal_question(question, context_analysis, prediction_results, patient_age=None, diabetes_duration=None):
        """Answer specific questions about the retinal analysis using RAG-enhanced responses"""

        # Initialize RAG service
        rag = initialize_rag()

        # Prepare patient context
        patient_info = ""
        if patient_age and diabetes_duration:
            patient_info = f"Patient: {patient_age} years old, diabetes for {diabetes_duration} years. "

        patient_context = f"{patient_info}Current AI Results: {prediction_results['class']} (Grade {prediction_results['value']}, {prediction_results['probability']:.2%} confidence)"

        # Use RAG-enhanced question answering if available
        if rag:
            try:
                return rag.enhanced_question_answering(
                    question=question,
                    previous_analysis=context_analysis,
                    patient_context=patient_context,
                    prediction_results=prediction_results
                )
            except Exception as e:
                print(f"Warning: RAG-enhanced answering failed, falling back to basic method: {e}")

        # Fallback to original method# ... [existing fallback code]
