#!/usr/bin/env python3
"""
Test script for the Geographic Information RAG System
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration...")
    try:
        from config import config
        print(f"✅ Configuration loaded successfully")
        print(f"   - API Key: {'✅ Set' if config.GOOGLE_API_KEY else '❌ Not Set'}")
        print(f"   - Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"   - Generation Model: {config.GENERATION_MODEL}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def test_utilities():
    """Test utility modules."""
    print("\n🛠️ Testing Utilities...")
    
    try:
        from utils.geo_utils import geo_utils
        print("✅ Geo utilities loaded successfully")
        
        # Test basic functionality
        point = geo_utils.create_point(40.7128, -74.0060)
        print(f"   - Point creation: ✅ ({point.x}, {point.y})")
        
        bbox = geo_utils.create_bounding_box(40.7128, -74.0060, 10)
        print(f"   - Bounding box creation: ✅")
        
    except Exception as e:
        print(f"❌ Geo utilities error: {str(e)}")
        return False
    
    try:
        from utils.data_utils import data_utils
        print("✅ Data utilities loaded successfully")
        
        # Test sample data creation
        cities_data = data_utils.create_sample_geographic_data()
        print(f"   - Sample cities data: ✅ ({len(cities_data)} records)")
        
        env_data = data_utils.create_environmental_data()
        print(f"   - Environmental data: ✅ ({len(env_data)} records)")
        
    except Exception as e:
        print(f"❌ Data utilities error: {str(e)}")
        return False
    
    return True

def test_rag_components():
    """Test RAG system components."""
    print("\n🤖 Testing RAG Components...")
    
    try:
        from rag.embedding_model import GeminiEmbeddingModel
        print("✅ Embedding model imported successfully")
    except Exception as e:
        print(f"❌ Embedding model import error: {str(e)}")
        return False
    
    try:
        from rag.vector_store import GeographicVectorStore
        print("✅ Vector store imported successfully")
    except Exception as e:
        print(f"❌ Vector store import error: {str(e)}")
        return False
    
    try:
        from rag.rag_system import GeographicRAGSystem
        print("✅ RAG system imported successfully")
    except Exception as e:
        print(f"❌ RAG system import error: {str(e)}")
        return False
    
    return True

def test_evaluation():
    """Test evaluation components."""
    print("\n📊 Testing Evaluation Components...")
    
    try:
        from evaluation.rag_evaluator import RAGEvaluator
        print("✅ RAG evaluator imported successfully")
    except Exception as e:
        print(f"❌ RAG evaluator import error: {str(e)}")
        return False
    
    return True

def test_streamlit_app():
    """Test Streamlit app import."""
    print("\n🌐 Testing Streamlit App...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test if we can import our app
        import app
        print("✅ App module imported successfully")
        
    except Exception as e:
        print(f"❌ Streamlit app error: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Geographic Information RAG System Tests...\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
        ("RAG Components", test_rag_components),
        ("Evaluation", test_evaluation),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the application:")
        print("1. Set your GOOGLE_API_KEY in the .env file")
        print("2. Run: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
