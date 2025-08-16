import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Union, Dict, Any
import numpy as np
import logging
from config import config

logger = logging.getLogger(__name__)

class GeminiEmbeddingModel:
    """Embedding model using Google Gemini for geographic data."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Gemini embedding model."""
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini embeddings")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize LangChain embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=self.api_key
        )
        
        logger.info("Gemini embedding model initialized successfully")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for texts: {str(e)}")
            raise
    
    def embed_geographic_data(self, data: Dict[str, Any]) -> List[float]:
        """Generate embedding for geographic data by combining relevant fields."""
        # Create a comprehensive text representation of geographic data
        text_parts = []
        
        if 'name' in data:
            text_parts.append(f"Location: {data['name']}")
        
        if 'description' in data:
            text_parts.append(f"Description: {data['description']}")
        
        if 'country' in data:
            text_parts.append(f"Country: {data['country']}")
        
        if 'landmarks' in data:
            text_parts.append(f"Landmarks: {data['landmarks']}")
        
        if 'climate' in data:
            text_parts.append(f"Climate: {data['climate']}")
        
        if 'type' in data:
            text_parts.append(f"Type: {data['type']}")
        
        if 'status' in data:
            text_parts.append(f"Status: {data['status']}")
        
        # Combine all text parts
        combined_text = " | ".join(text_parts)
        
        return self.embed_text(combined_text)
    
    def embed_satellite_data(self, satellite_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for satellite imagery data."""
        text_parts = []
        
        if 'description' in satellite_data:
            text_parts.append(f"Satellite Coverage: {satellite_data['description']}")
        
        if 'available_datasets' in satellite_data:
            datasets = ", ".join(satellite_data['available_datasets'])
            text_parts.append(f"Available Datasets: {datasets}")
        
        if 'cloud_cover_percent' in satellite_data:
            text_parts.append(f"Cloud Cover: {satellite_data['cloud_cover_percent']}%")
        
        if 'vegetation_index' in satellite_data:
            text_parts.append(f"Vegetation Index: {satellite_data['vegetation_index']}")
        
        if 'urban_area_percent' in satellite_data:
            text_parts.append(f"Urban Area: {satellite_data['urban_area_percent']}%")
        
        combined_text = " | ".join(text_parts)
        
        return self.embed_text(combined_text)
    
    def embed_spatial_query(self, query: str, coordinates: Dict[str, float] = None) -> List[float]:
        """Generate embedding for a spatial query."""
        query_parts = [query]
        
        if coordinates:
            lat = coordinates.get('latitude', coordinates.get('lat'))
            lon = coordinates.get('longitude', coordinates.get('lon'))
            if lat and lon:
                query_parts.append(f"Coordinates: {lat:.4f}, {lon:.4f}")
        
        combined_query = " | ".join(query_parts)
        
        return self.embed_text(combined_query)
    
    def calculate_similarity(self, embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def batch_embed_geographic_data(self, data_list: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for a batch of geographic data."""
        try:
            # Prepare text representations
            texts = []
            for data in data_list:
                text_parts = []
                
                if 'name' in data:
                    text_parts.append(f"Location: {data['name']}")
                
                if 'description' in data:
                    text_parts.append(f"Description: {data['description']}")
                
                if 'country' in data:
                    text_parts.append(f"Country: {data['country']}")
                
                if 'landmarks' in data:
                    text_parts.append(f"Landmarks: {data['landmarks']}")
                
                if 'climate' in data:
                    text_parts.append(f"Climate: {data['climate']}")
                
                if 'type' in data:
                    text_parts.append(f"Type: {data['type']}")
                
                if 'status' in data:
                    text_parts.append(f"Status: {data['status']}")
                
                combined_text = " | ".join(text_parts)
                texts.append(combined_text)
            
            # Generate embeddings in batch
            return self.embed_texts(texts)
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of the embeddings."""
        try:
            # Generate a test embedding to determine dimensions
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimensions: {str(e)}")
            return 768  # Default fallback dimension
