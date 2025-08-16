__import__('pysqlite3')
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import geopandas as gpd
import logging
import json
from config import config
from rag.embedding_model import GeminiEmbeddingModel
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


logger = logging.getLogger(__name__)


class GeographicVectorStore:
    """Vector store for geographic data using ChromaDB."""
    
    def __init__(self, persist_directory: str = None, embedding_model: GeminiEmbeddingModel = None):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.embedding_model = embedding_model or GeminiEmbeddingModel()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collections
        self.geographic_collection = self._get_or_create_collection("geographic_data")
        self.satellite_collection = self._get_or_create_collection("satellite_data")
        self.environmental_collection = self._get_or_create_collection("environmental_data")
        self.infrastructure_collection = self._get_or_create_collection("infrastructure_data")
        
        logger.info("Geographic vector store initialized successfully")
    
    def _get_or_create_collection(self, name: str):
        """Get existing collection or create a new one."""
        try:
            collection = self.client.get_collection(name=name)
            logger.info(f"Collection '{name}' loaded successfully")
        except:
            collection = self.client.create_collection(
                name=name,
                metadata={"description": f"Collection for {name}"}
            )
            logger.info(f"Collection '{name}' created successfully")
        
        return collection
    
    def add_geographic_data(self, data: gpd.GeoDataFrame, collection_name: str = "geographic_data") -> bool:
        """Add geographic data to the vector store."""
        try:
            # Convert GeoDataFrame to list of dictionaries
            data_list = data.to_dict('records')
            
            # Prepare data for vector store
            documents = []
            metadatas = []
            ids = []
            
            for i, record in enumerate(data_list):
                # Extract geometry information
                if 'geometry' in record:
                    geom = record['geometry']
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):
                        record['longitude'] = geom.x
                        record['latitude'] = geom.y
                    record.pop('geometry')  # Remove geometry object
                
                # Create document text
                doc_text = self._create_document_text(record)
                documents.append(doc_text)
                
                # Create metadata (preserve lat/lon for spatial search)
                metadata = {k: str(v) for k, v in record.items()}
                metadata['data_type'] = 'geographic'
                metadatas.append(metadata)
                
                # Create unique ID
                record_id = f"{collection_name}_{i}_{record.get('name', record.get('location', 'unknown'))}"
                ids.append(record_id)
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_texts(documents)
            
            # Add to collection
            collection = self._get_collection_by_name(collection_name)
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(data_list)} records to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding geographic data: {str(e)}")
            return False
    
    def add_satellite_data(self, satellite_data: List[Dict[str, Any]]) -> bool:
        """Add satellite imagery data to the vector store."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, data in enumerate(satellite_data):
                # Create document text
                doc_text = self._create_satellite_document_text(data)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {k: str(v) for k, v in data.items()}
                metadata['data_type'] = 'satellite'
                metadatas.append(metadata)
                
                # Create unique ID
                record_id = f"satellite_{i}_{data.get('location', {}).get('lat', 'unknown')}"
                ids.append(record_id)
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_texts(documents)
            
            # Add to satellite collection
            self.satellite_collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(satellite_data)} satellite data records")
            return True
            
        except Exception as e:
            logger.error(f"Error adding satellite data: {str(e)}")
            return False
    
    def _create_document_text(self, record: Dict[str, Any]) -> str:
        """Create a text document from a geographic record."""
        text_parts = []
        
        if 'name' in record:
            text_parts.append(f"Location: {record['name']}")
        if 'location' in record:
            text_parts.append(f"Location: {record['location']}")
        
        if 'description' in record:
            text_parts.append(f"Description: {record['description']}")
        
        if 'country' in record:
            text_parts.append(f"Country: {record['country']}")
        
        if 'landmarks' in record:
            text_parts.append(f"Landmarks: {record['landmarks']}")
        
        if 'climate' in record:
            text_parts.append(f"Climate: {record['climate']}")
        if 'climate_zone' in record:
            text_parts.append(f"Climate Zone: {record['climate_zone']}")
        
        if 'type' in record:
            text_parts.append(f"Type: {record['type']}")
        if 'infrastructure_type' in record:
            text_parts.append(f"Infrastructure: {record['infrastructure_type']}")
        
        if 'status' in record:
            text_parts.append(f"Status: {record['status']}")
        if 'completion_year' in record:
            text_parts.append(f"Completion Year: {record['completion_year']}")
        
        if 'population' in record:
            text_parts.append(f"Population: {record['population']}")
        if 'area_km2' in record:
            text_parts.append(f"Area (km²): {record['area_km2']}")
        if 'timezone' in record:
            text_parts.append(f"Timezone: {record['timezone']}")
        
        if 'air_quality' in record:
            text_parts.append(f"Air Quality: {record['air_quality']}")
        if 'air_quality_index' in record:
            text_parts.append(f"AQI: {record['air_quality_index']}")
        if 'water_quality' in record:
            text_parts.append(f"Water Quality: {record['water_quality']}")
        if 'annual_rainfall_mm' in record:
            text_parts.append(f"Annual Rainfall (mm): {record['annual_rainfall_mm']}")
        if 'temperature_range_c' in record:
            text_parts.append(f"Temperature Range (°C): {record['temperature_range_c']}")
        
        if 'vegetation_coverage' in record:
            text_parts.append(f"Vegetation Coverage: {record['vegetation_coverage']}")
        
        if 'capacity' in record:
            text_parts.append(f"Capacity: {record['capacity']}")
        if 'maintenance_status' in record:
            text_parts.append(f"Maintenance: {record['maintenance_status']}")
        
        return " | ".join(text_parts)
    
    def _create_satellite_document_text(self, data: Dict[str, Any]) -> str:
        """Create a text document from satellite data."""
        text_parts = []
        
        if 'description' in data:
            text_parts.append(f"Satellite Coverage: {data['description']}")
        
        if 'available_datasets' in data:
            datasets = ", ".join(data['available_datasets'])
            text_parts.append(f"Available Datasets: {datasets}")
        
        if 'cloud_cover_percent' in data:
            text_parts.append(f"Cloud Cover: {data['cloud_cover_percent']}%")
        
        if 'vegetation_index' in data:
            text_parts.append(f"Vegetation Index: {data['vegetation_index']}")
        
        if 'urban_area_percent' in data:
            text_parts.append(f"Urban Area: {data['urban_area_percent']}%")
        
        if 'water_bodies_detected' in data:
            text_parts.append(f"Water Bodies: {data['water_bodies_detected']}")
        
        return " | ".join(text_parts)
    
    def _get_collection_by_name(self, name: str):
        """Get a collection by name."""
        collections = {
            "geographic_data": self.geographic_collection,
            "satellite_data": self.satellite_collection,
            "environmental_data": self.environmental_collection,
            "infrastructure_data": self.infrastructure_collection
        }
        return collections.get(name, self.geographic_collection)
    
    def search(self, query: str, collection_name: str = "geographic_data", 
               n_results: int = None, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar geographic data."""
        try:
            n_results = n_results or config.TOP_K_RETRIEVAL
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_spatial_query(query)
            
            # Get collection
            collection = self._get_collection_by_name(collection_name)
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Search completed with {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
    
    def spatial_search(self, query: str, lat: float, lon: float, 
                      radius_km: float = 10, n_results: int = None) -> List[Dict[str, Any]]:
        """Perform spatial search with geographic constraints."""
        try:
            n_results = n_results or config.TOP_K_RETRIEVAL
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_spatial_query(
                query, {'latitude': lat, 'longitude': lon}
            )
            
            # Search in all collections
            all_results = []
            
            for collection_name in ["geographic_data", "environmental_data", "infrastructure_data"]:
                collection = self._get_collection_by_name(collection_name)
                
                # Get all documents in collection
                all_docs = collection.get()
                
                # Filter by spatial proximity
                spatial_results = []
                for i, metadata in enumerate(all_docs['metadatas']):
                    if 'longitude' in metadata and 'latitude' in metadata:
                        try:
                            doc_lat = float(metadata['latitude'])
                            doc_lon = float(metadata['longitude'])
                            
                            # Calculate distance
                            distance = self._calculate_distance(lat, lon, doc_lat, doc_lon)
                            
                            if distance <= radius_km:
                                spatial_results.append({
                                    'document': all_docs['documents'][i],
                                    'metadata': metadata,
                                    'distance_km': distance,
                                    'id': all_docs['ids'][i],
                                    'collection': collection_name
                                })
                        except (ValueError, TypeError):
                            continue
                
                # Sort by distance and take top results
                spatial_results.sort(key=lambda x: x['distance_km'])
                all_results.extend(spatial_results[:n_results])
            
            # Sort all results by distance
            all_results.sort(key=lambda x: x['distance_km'])
            
            logger.info(f"Spatial search completed with {len(all_results)} results")
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in spatial search: {str(e)}")
            return []
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        from geopy.distance import geodesic
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        stats = {}
        
        for collection_name in ["geographic_data", "satellite_data", "environmental_data", "infrastructure_data"]:
            try:
                collection = self._get_collection_by_name(collection_name)
                count = collection.count()
                stats[collection_name] = count
            except Exception as e:
                stats[collection_name] = 0
                logger.warning(f"Could not get count for collection {collection_name}: {str(e)}")
        
        return stats
    
    def clear_collection(self, collection_name: str) -> bool:
        """Clear a specific collection."""
        try:
            collection = self._get_collection_by_name(collection_name)
            collection.delete()
            
            # Recreate the collection
            self._get_or_create_collection(collection_name)
            
            logger.info(f"Collection '{collection_name}' cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {str(e)}")
            return False
    
    def export_collection_data(self, collection_name: str) -> pd.DataFrame:
        """Export collection data to a DataFrame."""
        try:
            collection = self._get_collection_by_name(collection_name)
            data = collection.get()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'id': data['ids'],
                'document': data['documents'],
                'metadata': data['metadatas']
            })
            
            # Expand metadata
            metadata_df = pd.json_normalize(df['metadata'])
            df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting collection data: {str(e)}")
            return pd.DataFrame()
