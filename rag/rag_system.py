import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any
import logging
import time
from config import config
from rag.vector_store import GeographicVectorStore
from rag.embedding_model import GeminiEmbeddingModel
from utils.geo_utils import geo_utils
from utils.data_utils import data_utils

logger = logging.getLogger(__name__)

class GeographicRAGSystem:
    """Main RAG system for geographic information retrieval and generation."""
    
    def __init__(self, api_key: str = None):
        """Initialize the RAG system."""
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini LLM")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize components
        self.embedding_model = GeminiEmbeddingModel(self.api_key)
        self.vector_store = GeographicVectorStore(embedding_model=self.embedding_model)
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.GENERATION_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize system prompt
        self.system_prompt = self._create_system_prompt()
        
        logger.info("Geographic RAG system initialized successfully")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM."""
        return """You are a Geographic Information System (GIS) and Satellite Imagery expert assistant. Your role is to:

1. Analyze geographic data and provide accurate, location-specific insights
2. Interpret satellite imagery and environmental data with technical precision
3. Answer spatial queries with precise geographic context
4. Provide recommendations based on geographic and satellite analysis
5. Explain complex spatial relationships and remote sensing concepts in simple terms
6. Analyze multi-spectral satellite data for land use, vegetation, and environmental monitoring
7. Interpret NDVI, urban heat island effects, and change detection from satellite imagery

When responding:
- Always provide geographic context (coordinates, location names, distances)
- Use precise spatial and remote sensing terminology
- Include relevant environmental, infrastructure, and satellite-derived information
- Suggest additional geographic and satellite analysis when appropriate
- Cite specific data sources and sensor types when available
- Explain satellite imagery resolution, bands, and analysis capabilities

Focus on accuracy, relevance, and actionable insights for geographic decision-making and environmental monitoring."""

    def query(self, user_query: str, coordinates: Dict[str, float] = None, 
              radius_km: float = None, collection_name: str = "geographic_data") -> Dict[str, Any]:
        """Process a user query using the RAG system."""
        try:
            start_time = time.time()
            
            # Step 1: Retrieve relevant documents
            if coordinates and radius_km:
                # Perform spatial search
                retrieved_docs = self.vector_store.spatial_search(
                    user_query, 
                    coordinates.get('latitude', coordinates.get('lat')), 
                    coordinates.get('longitude', coordinates.get('lon')), 
                    radius_km
                )
            else:
                # Perform regular semantic search
                retrieved_docs = self.vector_store.search(user_query, collection_name)
            
            # Step 2: Prepare context for LLM
            context = self._prepare_context(retrieved_docs)
            
            # Step 3: Generate response using LLM
            response = self._generate_response(user_query, context, coordinates)
            
            # Step 4: Calculate metrics
            processing_time = time.time() - start_time
            
            result = {
                'query': user_query,
                'response': response,
                'retrieved_documents': retrieved_docs,
                'context_summary': context,
                'processing_time': processing_time,
                'coordinates': coordinates,
                'search_radius_km': radius_km,
                'collection_used': collection_name
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': user_query,
                'error': str(e),
                'response': "I apologize, but I encountered an error processing your query. Please try again."
            }
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents for the LLM."""
        if not retrieved_docs:
            return "No relevant geographic data found for the query."
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Limit to top 3 documents
            context_parts.append(f"Document {i+1}:")
            context_parts.append(f"Content: {doc['document']}")
            
            if 'metadata' in doc:
                metadata = doc['metadata']
                if 'name' in metadata:
                    context_parts.append(f"Location: {metadata['name']}")
                if 'latitude' in metadata and 'longitude' in metadata:
                    context_parts.append(f"Coordinates: {metadata['latitude']}, {metadata['longitude']}")
                if 'distance_km' in doc:
                    context_parts.append(f"Distance: {doc['distance_km']:.2f} km")
            
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, 
                          coordinates: Dict[str, float] = None) -> str:
        """Generate response using the LLM with retrieved context."""
        try:
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._create_user_prompt(query, context, coordinates))
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"I apologize, but I encountered an error generating a response. Error: {str(e)}"
    
    def _create_user_prompt(self, query: str, context: str, 
                           coordinates: Dict[str, float] = None) -> str:
        """Create the user prompt for the LLM."""
        prompt_parts = [
            f"User Query: {query}",
            "",
            "Retrieved Geographic Context:",
            context,
            ""
        ]
        
        if coordinates:
            lat = coordinates.get('latitude', coordinates.get('lat'))
            lon = coordinates.get('longitude', coordinates.get('lon'))
            prompt_parts.append(f"Query Location: Latitude {lat}, Longitude {lon}")
            prompt_parts.append("")
        
        prompt_parts.append(
            "Please provide a comprehensive answer based on the retrieved geographic context. "
            "Include specific location details, spatial relationships, and actionable insights. "
            "If the context is insufficient, suggest what additional geographic data might be helpful."
        )
        
        return "\n".join(prompt_parts)
    
    def analyze_location(self, lat: float, lon: float, 
                        radius_km: float = 10) -> Dict[str, Any]:
        """Perform comprehensive analysis of a specific location."""
        try:
            # Get geographic data in the area
            geographic_data = self.vector_store.spatial_search(
                "location analysis", lat, lon, radius_km
            )
            
            # Get satellite imagery information
            satellite_info = data_utils.fetch_satellite_imagery_info(lat, lon, radius_km)
            processed_satellite = data_utils.process_satellite_data(satellite_info)
            
            # Create bounding box for visualization
            bounding_box = geo_utils.create_bounding_box(lat, lon, radius_km)
            
            # Generate comprehensive analysis
            analysis_query = f"""
            Analyze the geographic area around coordinates {lat:.4f}, {lon:.4f} 
            within a {radius_km} km radius. Consider:
            1. Geographic features and landmarks
            2. Environmental conditions
            3. Infrastructure and development
            4. Spatial relationships and patterns
            5. Recommendations for further analysis
            """
            
            analysis_result = self.query(analysis_query, {'latitude': lat, 'longitude': lon}, radius_km)
            
            return {
                'coordinates': {'latitude': lat, 'longitude': lon},
                'radius_km': radius_km,
                'geographic_data': geographic_data,
                'satellite_analysis': processed_satellite,
                'bounding_box': bounding_box,
                'comprehensive_analysis': analysis_result,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing location: {str(e)}")
            return {
                'error': str(e),
                'coordinates': {'latitude': lat, 'longitude': lon}
            }
    
    def compare_locations(self, location1: Dict[str, float], 
                         location2: Dict[str, float]) -> Dict[str, Any]:
        """Compare two geographic locations."""
        try:
            # Analyze both locations
            analysis1 = self.analyze_location(
                location1['latitude'], location1['longitude']
            )
            analysis2 = self.analyze_location(
                location2['latitude'], location2['longitude']
            )
            
            # Calculate distance between locations
            distance = geo_utils.calculate_distance(
                (location1['latitude'], location1['longitude']),
                (location2['latitude'], location2['longitude'])
            )
            
            # Generate comparison
            comparison_query = f"""
            Compare these two geographic locations:
            Location 1: {location1['latitude']:.4f}, {location1['longitude']:.4f}
            Location 2: {location2['latitude']:.4f}, {location2['longitude']:.4f}
            Distance between them: {distance:.2f} km
            
            Provide a detailed comparison covering:
            1. Geographic similarities and differences
            2. Environmental conditions
            3. Infrastructure and development patterns
            4. Spatial relationships
            5. Potential implications for planning or analysis
            """
            
            comparison_result = self.query(comparison_query)
            
            return {
                'location1': analysis1,
                'location2': analysis2,
                'distance_km': distance,
                'comparison_analysis': comparison_result,
                'comparison_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error comparing locations: {str(e)}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system."""
        try:
            # Get vector store statistics
            collection_stats = self.vector_store.get_collection_stats()
            
            # Test LLM connectivity
            try:
                test_response = self.llm.invoke([HumanMessage(content="Test")])
                llm_status = "Connected"
            except Exception as e:
                llm_status = f"Error: {str(e)}"
            
            # Test embedding model
            try:
                test_embedding = self.embedding_model.embed_text("Test")
                embedding_status = "Connected"
                embedding_dimensions = len(test_embedding)
            except Exception as e:
                embedding_status = f"Error: {str(e)}"
                embedding_dimensions = 0
            
            return {
                'system_status': 'Operational',
                'timestamp': time.time(),
                'llm_status': llm_status,
                'embedding_status': embedding_status,
                'embedding_dimensions': embedding_dimensions,
                'collection_statistics': collection_stats,
                'total_documents': sum(collection_stats.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'system_status': 'Error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def export_analysis_results(self, analysis_result: Dict[str, Any], 
                               format_type: str = 'json') -> str:
        """Export analysis results to various formats."""
        try:
            if format_type == 'json':
                import json
                return json.dumps(analysis_result, indent=2, default=str)
            
            elif format_type == 'csv':
                # Convert to DataFrame and export
                import pandas as pd
                
                # Flatten the analysis result
                flattened_data = []
                self._flatten_dict(analysis_result, flattened_data)
                
                df = pd.DataFrame(flattened_data)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting analysis results: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def _flatten_dict(self, data: Any, result: List[Dict[str, Any]], 
                      prefix: str = "") -> None:
        """Recursively flatten a nested dictionary for CSV export."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}.{key}" if prefix else key
                self._flatten_dict(value, result, new_key)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                self._flatten_dict(item, result, new_key)
        else:
            result.append({"key": prefix, "value": str(data)})

    def get_collection_counts(self) -> Dict[str, int]:
        """Return counts of items per collection in the vector store."""
        try:
            return self.vector_store.get_collection_stats()
        except Exception as e:
            logger.warning(f"Could not fetch collection stats: {e}")
            return {}

    def ingest_sample_data(self, cities_df=None, environmental_gdf=None, infrastructure_gdf=None, satellite_gdf=None) -> Dict[str, bool]:
        """Ingest provided dataframes into their respective vector store collections."""
        results: Dict[str, bool] = {}
        try:
            if cities_df is not None and len(cities_df) > 0:
                results['geographic_data'] = self.vector_store.add_geographic_data(cities_df, 'geographic_data')
            if environmental_gdf is not None and len(environmental_gdf) > 0:
                results['environmental_data'] = self.vector_store.add_geographic_data(environmental_gdf, 'environmental_data')
            if infrastructure_gdf is not None and len(infrastructure_gdf) > 0:
                results['infrastructure_data'] = self.vector_store.add_geographic_data(infrastructure_gdf, 'infrastructure_data')
            if satellite_gdf is not None and len(satellite_gdf) > 0:
                results['satellite_data'] = self.vector_store.add_geographic_data(satellite_gdf, 'satellite_data')
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
        return results
