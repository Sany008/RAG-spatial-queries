import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Geographic Information RAG system."""
    
    # Gemini API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database Settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Geographic Data Settings
    DEFAULT_CRS: str = os.getenv("DEFAULT_CRS", "EPSG:4326")
    MAX_QUERY_RADIUS_KM: float = float(os.getenv("MAX_QUERY_RADIUS_KM", "100"))
    
    # API Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    
    # Model Configuration
    EMBEDDING_MODEL: str = "models/embedding-001"
    GENERATION_MODEL: str = "gemini-1.5-flash"
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your .env file.")
        return True
    
    @classmethod
    def get_chroma_settings(cls) -> dict:
        """Get ChromaDB settings."""
        return {
            "persist_directory": cls.CHROMA_PERSIST_DIRECTORY,
            "anonymized_telemetry": False
        }

# Global configuration instance
config = Config()
