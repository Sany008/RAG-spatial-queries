# 🌍 Geographic Information RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that combines geographic data, satellite imagery, and location-based information to answer spatial queries and provide location-specific insights. Built with modern AI technologies including Google Gemini LLM, ChromaDB vector database, and advanced geospatial analysis capabilities.

## 🚀 Features

### Core RAG Capabilities
- **Intelligent Spatial Queries**: Natural language queries with geographic context
- **Multi-Modal Data Integration**: Text, coordinates, and satellite imagery analysis
- **Vector Database Search**: ChromaDB-powered semantic and spatial search
- **Context-Aware Generation**: Gemini LLM with retrieved geographic context

### Geographic Data Processing
- **Spatial Indexing**: Efficient geographic coordinate systems and spatial relationships
- **Multi-Scale Analysis**: City-level to regional geographic insights
- **Data Validation**: Quality checks and coordinate validation
- **Export Capabilities**: Multiple format support (GeoJSON, CSV, Shapefile)

### Satellite Imagery Analysis
- **Multi-Sensor Support**: Sentinel-2, Landsat-8/9, PlanetScope, MODIS
- **Environmental Monitoring**: NDVI analysis, vegetation health, urban expansion
- **Change Detection**: Temporal analysis and trend identification
- **Land Use Classification**: Urban areas, vegetation, water bodies, agricultural land

### Interactive Visualization
- **Interactive Maps**: Folium-based geographic visualization
- **Real-time Analysis**: Dynamic satellite imagery analysis
- **Time Series Charts**: NDVI trends, urban expansion patterns
- **Professional Reports**: Comprehensive analysis with recommendations

### India-Focused Geographic Data
- **Major Cities**: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata
- **Environmental Data**: Air quality, water quality, climate zones, vegetation coverage
- **Infrastructure Projects**: Metro systems, airports, ports, smart city initiatives
- **Satellite Monitoring**: Coastal erosion, urban development, environmental impact

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   RAG System    │    │  Vector Store   │
│                 │    │                 │    │   (ChromaDB)    │
│ • Query Input   │◄──►│ • Query Process │◄──►│ • Geographic    │
│ • Map Display   │    │ • Context Prep  │    │ • Environmental │
│ • Analysis      │    │ • LLM Gen       │    │ • Infrastructure│
│ • Evaluation    │    │ • Response Gen  │    │ • Satellite     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Gemini LLM     │              │
         │              │  + Embeddings   │              │
         └──────────────┼─────────────────┼──────────────┘
                        │ • Text Gen      │
                        │ • Embeddings   │
                        │ • Analysis     │
                        └─────────────────┘
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.13+**: Modern Python with advanced features
- **Streamlit**: Interactive web application framework
- **LangChain**: RAG framework and LLM orchestration
- **Google Gemini**: Advanced language model and embeddings

### Vector Database & Search
- **ChromaDB**: High-performance vector database
- **Semantic Search**: Context-aware document retrieval
- **Spatial Search**: Coordinate-based geographic queries
- **Multi-Collection**: Organized data storage by category

### Geospatial Libraries
- **GeoPandas**: Geographic data manipulation
- **Shapely**: Geometric operations and spatial analysis
- **Folium**: Interactive mapping and visualization
- **PyProj**: Coordinate system transformations

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **PIL/Pillow**: Image processing capabilities
- **Requests**: HTTP client for API interactions

### Evaluation & Monitoring
- **DeepEval**: RAG system performance metrics
- **Custom Metrics**: Answer relevancy, context recall, faithfulness
- **Performance Reports**: Comprehensive evaluation summaries
- **Export Functionality**: Detailed analysis reports

## 📋 Prerequisites

### System Requirements
- **Python**: 3.13 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and data
- **OS**: macOS, Linux, or Windows (with WSL)

### API Keys Required
- **Google Gemini API Key**: For LLM and embeddings
- **Optional**: Additional satellite imagery APIs for production use

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd project-18
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API key
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 🔧 Detailed Setup

### Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### API Key Setup
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file
4. Restart the application

### Virtual Environment Management
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Deactivate
deactivate

# Update dependencies
python -m pip install -r requirements.txt --upgrade
```

## 📖 Usage Guide

### 1. Home Page
- **System Status**: Check if all components are operational
- **Feature Overview**: Understand system capabilities
- **India-Focused Data**: Explore geographic data coverage

### 2. Query System
- **Natural Language Queries**: Ask questions about geographic locations
- **Coordinate Selection**: Use interactive map for precise location selection
- **Automatic Collection Detection**: System automatically selects relevant data
- **Example Queries**:
  - "What is the air quality in Delhi?"
  - "Show me satellite imagery of Mumbai"
  - "Analyze land use changes in Bangalore"

### 3. Interactive Maps
- **Data Visualization**: View cities, environmental data, and infrastructure
- **Interactive Markers**: Click for detailed information
- **Multiple Data Types**: Switch between different geographic datasets

### 4. Satellite Imagery Analysis
- **Location Selection**: Choose from major Indian cities
- **Sensor Selection**: Sentinel-2, Landsat-8/9, PlanetScope, MODIS
- **Temporal Analysis**: Set date ranges for change detection
- **Environmental Metrics**: NDVI, urban expansion, vegetation health

### 5. Data Analysis
- **Statistical Overview**: Data counts and summaries
- **Interactive Tables**: Explore raw data with filtering
- **Export Options**: Download data in multiple formats

### 6. System Evaluation
- **Performance Metrics**: Answer relevancy, context recall, faithfulness
- **Test Cases**: Comprehensive evaluation scenarios
- **Performance Reports**: Detailed analysis summaries
- **Export Results**: Save evaluation reports

## 🔍 Example Queries

### Geographic Queries
```
"What is the population of Mumbai?"
"Tell me about Delhi Metro infrastructure"
"How is the climate in Bangalore?"
"What are the main ports in Chennai?"
```

### Satellite Imagery Queries
```
"Show me satellite imagery of Delhi"
"What is the NDVI value in Mumbai?"
"Analyze land use changes in Bangalore"
"Detect urban expansion in Chennai"
"Monitor coastal erosion in Mumbai"
```

### Environmental Queries
```
"What is the air quality in Delhi?"
"Show me water quality data for Mumbai"
"Analyze vegetation coverage in Bangalore"
"What are the environmental concerns in Kolkata?"
```

## 📊 Data Collections

### Geographic Data
- **Cities**: Population, area, coordinates, timezone
- **Global Coverage**: Major cities worldwide
- **India Focus**: Comprehensive Indian city data

### Environmental Data
- **Air Quality**: AQI values and pollution levels
- **Water Quality**: Assessment and monitoring data
- **Climate Zones**: Temperature ranges and rainfall
- **Vegetation**: Coverage percentages and health indices

### Infrastructure Data
- **Transportation**: Metro systems, airports, ports
- **Smart Cities**: Development initiatives and projects
- **Construction**: Completion dates and status
- **Maintenance**: Current condition and capacity

### Satellite Imagery Data
- **Multi-Sensor**: Various satellite platforms
- **Resolution**: 3m to 100m coverage
- **Temporal**: Time series analysis capabilities
- **Environmental**: NDVI, land use, change detection

## 🧪 Testing & Evaluation

### Automated Testing
```bash
# Run system tests
python test_system.py

# Run specific component tests
python -c "from utils.data_utils import data_utils; print('Data utils working')"
```

### Performance Evaluation
- **DeepEval Integration**: Professional RAG evaluation metrics
- **Custom Test Cases**: Geographic and satellite-focused scenarios
- **Performance Reports**: Comprehensive evaluation summaries
- **Export Functionality**: Save results for analysis

### Evaluation Metrics
- **Answer Relevancy**: Response quality assessment
- **Context Relevancy**: Retrieved information appropriateness
- **Context Recall**: Information retrieval completeness
- **Faithfulness**: Response accuracy to source material

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit deploy app.py

# Using Docker
docker build -t geographic-rag .
docker run -p 8501:8501 geographic-rag

# Using Heroku
heroku create your-app-name
git push heroku main
```

### Environment Configuration
```bash
# Production environment variables
export GOOGLE_API_KEY="your_production_key"
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check environment variable
echo $GOOGLE_API_KEY

# Verify .env file
cat .env
```

#### 2. Dependency Issues
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python version
python --version
```

#### 3. Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Port Conflicts
```bash
# Use different port
streamlit run app.py --server.port 8502

# Check port usage
lsof -i :8501
```

### Performance Optimization
- **Vector Database**: Ensure adequate memory for ChromaDB
- **API Rate Limits**: Monitor Gemini API usage
- **Data Caching**: Implement caching for frequently accessed data
- **Async Processing**: Use background tasks for heavy operations

## 📈 Future Enhancements

### Planned Features
- **Real-time Satellite Data**: Live imagery and monitoring
- **Advanced Analytics**: Machine learning-based insights
- **Mobile Application**: Cross-platform mobile support
- **API Endpoints**: RESTful API for external integrations
- **Multi-language Support**: Internationalization capabilities

### Scalability Improvements
- **Distributed Processing**: Multi-node deployment
- **Cloud Integration**: AWS, GCP, Azure support
- **Database Optimization**: Advanced indexing and caching
- **Load Balancing**: High-availability deployment

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Code Standards
- **Python**: Follow PEP 8 guidelines
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Type Hints**: Use type annotations

### Testing Guidelines
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest test_system.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: Advanced language model capabilities
- **ChromaDB**: High-performance vector database
- **Streamlit**: Interactive web application framework
- **DeepEval**: RAG system evaluation framework
- **Open Source Community**: Various geospatial and AI libraries

## 📞 Support

### Getting Help
- **Issues**: Create GitHub issues for bugs and feature requests
- **Documentation**: Check this README and inline code comments
- **Community**: Join our discussion forum
- **Email**: Contact the development team

### Useful Resources
- [Google Gemini Documentation](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [DeepEval Documentation](https://docs.confident-ai.com)

---

**Built with ❤️ for Geographic Information Science and AI Innovation**

*Last updated: August 2024*
