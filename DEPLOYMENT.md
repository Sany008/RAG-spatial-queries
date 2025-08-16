# Deployment Guide for Geographic Information RAG System

This guide provides step-by-step instructions for deploying the Geographic Information RAG system.

## 🚀 Quick Deployment

### 1. Local Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd project-18

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env and add your Gemini API key
```

### 2. Environment Configuration

Create a `.env` file with:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
DEBUG=False
LOG_LEVEL=INFO
CHROMA_PERSIST_DIRECTORY=./chroma_db
DEFAULT_CRS=EPSG:4326
MAX_QUERY_RADIUS_KM=100
MAX_REQUESTS_PER_MINUTE=60
```

### 3. Run the Application

```bash
# Test the system
python test_system.py

# Run the Streamlit app
streamlit run app.py
```

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set environment variables:
     - `GOOGLE_API_KEY`: Your Gemini API key
   - Deploy

### Option 2: Heroku

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**
   ```
   python-3.9.18
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   heroku config:set GOOGLE_API_KEY=your_api_key
   git push heroku main
   ```

### Option 3: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**
   ```bash
   docker build -t geo-rag-system .
   docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key geo-rag-system
   ```

## 🔧 Production Configuration

### Environment Variables

```bash
# Production settings
DEBUG=False
LOG_LEVEL=WARNING
MAX_REQUESTS_PER_MINUTE=100
CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db

# Security
GOOGLE_API_KEY=your_production_api_key
```

### Performance Optimization

1. **Vector Database**
   - Use persistent storage for ChromaDB
   - Consider using external vector databases (Pinecone, Weaviate) for production

2. **Caching**
   - Implement Redis for response caching
   - Cache frequently requested geographic data

3. **Rate Limiting**
   - Implement proper rate limiting for API calls
   - Use queue systems for heavy processing

## 📊 Monitoring and Logging

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    }
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify GOOGLE_API_KEY is set correctly
   - Check API key permissions and quotas

2. **Memory Issues**
   - Reduce chunk size for large datasets
   - Implement data pagination

3. **Performance Issues**
   - Monitor response times
   - Check vector database performance
   - Optimize embedding generation

### Debug Mode

```bash
# Enable debug mode
DEBUG=True
LOG_LEVEL=DEBUG

# Run with verbose logging
streamlit run app.py --logger.level=debug
```

## 🔒 Security Considerations

1. **API Key Management**
   - Never commit API keys to version control
   - Use environment variables or secret management
   - Rotate keys regularly

2. **Data Privacy**
   - Implement data access controls
   - Anonymize sensitive geographic data
   - Comply with data protection regulations

3. **Network Security**
   - Use HTTPS in production
   - Implement proper authentication
   - Restrict access to admin functions

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancing**
   - Use multiple Streamlit instances
   - Implement reverse proxy (nginx)

2. **Database Scaling**
   - Use external vector databases
   - Implement database sharding

3. **Caching Layer**
   - Redis for response caching
   - CDN for static assets

### Vertical Scaling

1. **Resource Allocation**
   - Increase memory allocation
   - Use more powerful CPUs
   - Optimize batch processing

## 🎯 Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] API keys secured
- [ ] Database configured
- [ ] Logging set up
- [ ] Health checks implemented
- [ ] Monitoring configured
- [ ] Security measures in place
- [ ] Performance tested
- [ ] Documentation updated

## 📞 Support

For deployment issues:
1. Check the logs for error messages
2. Verify environment configuration
3. Test locally before deploying
4. Check system requirements
5. Review security settings

## 🔄 Updates and Maintenance

1. **Regular Updates**
   - Update dependencies monthly
   - Monitor for security patches
   - Update API keys as needed

2. **Backup Strategy**
   - Backup vector database regularly
   - Version control for configuration
   - Document changes and updates

3. **Performance Monitoring**
   - Track response times
   - Monitor resource usage
   - Set up alerts for issues
