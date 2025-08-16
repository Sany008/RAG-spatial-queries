import pandas as pd
import geopandas as gpd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import requests
from PIL import Image
import io
import json
import os
from shapely.geometry import Point, Polygon
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class DataUtils:
    """Utility class for data processing and management."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.geojson', '.shp', '.json']
        self.satellite_sensors = [
            'Sentinel-2', 'Landsat-8', 'Landsat-9', 'MODIS', 'PlanetScope',
            'RapidEye', 'SPOT', 'IKONOS', 'QuickBird', 'WorldView'
        ]
    
    def load_geographic_data(self, file_path: str) -> gpd.GeoDataFrame:
        """Load geographic data from various file formats."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                # Load CSV and convert to GeoDataFrame
                df = pd.read_csv(file_path)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
                    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                else:
                    raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
            
            elif file_ext == '.geojson':
                return gpd.read_file(file_path)
            
            elif file_ext == '.shp':
                return gpd.read_file(file_path)
            
            elif file_ext == '.json':
                return gpd.read_file(file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def create_sample_geographic_data(self) -> pd.DataFrame:
        """Create sample geographic data for demonstration."""
        cities_data = [
            # Global cities
            {"name": "New York", "country": "USA", "latitude": 40.7128, "longitude": -74.0060, "population": 8336817, "area_km2": 778.2, "timezone": "UTC-5"},
            {"name": "London", "country": "UK", "latitude": 51.5074, "longitude": -0.1278, "population": 8982000, "area_km2": 1572, "timezone": "UTC+0"},
            {"name": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503, "population": 13929286, "area_km2": 2194, "timezone": "UTC+9"},
            {"name": "Sydney", "country": "Australia", "latitude": -33.8688, "longitude": 151.2093, "population": 5312163, "area_km2": 12367.7, "timezone": "UTC+10"},
            {"name": "Paris", "country": "France", "latitude": 48.8566, "longitude": 2.3522, "population": 2161000, "area_km2": 105.4, "timezone": "UTC+1"},
            
            # Indian cities
            {"name": "Mumbai", "country": "India", "latitude": 19.0760, "longitude": 72.8777, "population": 20411274, "area_km2": 603.4, "timezone": "UTC+5:30"},
            {"name": "Delhi", "country": "India", "latitude": 28.7041, "longitude": 77.1025, "population": 16787941, "area_km2": 1484, "timezone": "UTC+5:30"},
            {"name": "Bangalore", "country": "India", "latitude": 12.9716, "longitude": 77.5946, "population": 12425304, "area_km2": 741, "timezone": "UTC+5:30"},
            {"name": "Hyderabad", "country": "India", "latitude": 17.3850, "longitude": 78.4867, "population": 10469000, "area_km2": 650, "timezone": "UTC+5:30"},
            {"name": "Chennai", "country": "India", "latitude": 13.0827, "longitude": 80.2707, "population": 7088000, "area_km2": 426, "timezone": "UTC+5:30"},
            {"name": "Kolkata", "country": "India", "latitude": 22.5726, "longitude": 88.3639, "population": 14916388, "area_km2": 185, "timezone": "UTC+5:30"},
            {"name": "Pune", "country": "India", "latitude": 18.5204, "longitude": 73.8567, "population": 3124458, "area_km2": 331.26, "timezone": "UTC+5:30"},
            {"name": "Ahmedabad", "country": "India", "latitude": 23.0225, "longitude": 72.5714, "population": 5570585, "area_km2": 464, "timezone": "UTC+5:30"},
            {"name": "Jaipur", "country": "India", "latitude": 26.9124, "longitude": 75.7873, "population": 3073350, "area_km2": 467, "timezone": "UTC+5:30"},
            {"name": "Surat", "country": "India", "latitude": 21.1702, "longitude": 72.8311, "population": 4467797, "area_km2": 326.515, "timezone": "UTC+5:30"}
        ]
        
        df = pd.DataFrame(cities_data)
        df['city_id'] = range(1, len(df) + 1)
        return df
    
    def create_environmental_data(self) -> gpd.GeoDataFrame:
        """Create sample environmental data for demonstration."""
        env_data = [
            # Global environmental data
            {
                'location': 'New York',
                'country': 'USA',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'air_quality_index': 45,
                'water_quality': 'Good',
                'vegetation_coverage': 0.35,
                'climate_zone': 'Humid Subtropical',
                'annual_rainfall_mm': 1200,
                'temperature_range_c': '(-5, 30)'
            },
            {
                'location': 'London',
                'country': 'UK',
                'latitude': 51.5074,
                'longitude': -0.1278,
                'air_quality_index': 52,
                'water_quality': 'Good',
                'vegetation_coverage': 0.28,
                'climate_zone': 'Temperate Maritime',
                'annual_rainfall_mm': 600,
                'temperature_range_c': '(2, 25)'
            },
            {
                'location': 'Tokyo',
                'country': 'Japan',
                'latitude': 35.6762,
                'longitude': 139.6503,
                'air_quality_index': 38,
                'water_quality': 'Excellent',
                'vegetation_coverage': 0.42,
                'climate_zone': 'Humid Subtropical',
                'annual_rainfall_mm': 1500,
                'temperature_range_c': '(-2, 35)'
            },
            
            # Indian environmental data
            {
                'location': 'Mumbai',
                'country': 'India',
                'latitude': 19.0760,
                'longitude': 72.8777,
                'air_quality_index': 85,
                'water_quality': 'Moderate',
                'vegetation_coverage': 0.25,
                'climate_zone': 'Tropical Wet and Dry',
                'annual_rainfall_mm': 2500,
                'temperature_range_c': '(16, 38)'
            },
            {
                'location': 'Delhi',
                'country': 'India',
                'latitude': 28.7041,
                'longitude': 77.1025,
                'air_quality_index': 120,
                'water_quality': 'Poor',
                'vegetation_coverage': 0.18,
                'climate_zone': 'Semi-arid',
                'annual_rainfall_mm': 800,
                'temperature_range_c': '(5, 45)'
            },
            {
                'location': 'Bangalore',
                'country': 'India',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'air_quality_index': 65,
                'water_quality': 'Good',
                'vegetation_coverage': 0.40,
                'climate_zone': 'Tropical Savanna',
                'annual_rainfall_mm': 900,
                'temperature_range_c': '(15, 35)'
            },
            {
                'location': 'Chennai',
                'country': 'India',
                'latitude': 13.0827,
                'longitude': 80.2707,
                'air_quality_index': 75,
                'water_quality': 'Moderate',
                'vegetation_coverage': 0.22,
                'climate_zone': 'Tropical Wet and Dry',
                'annual_rainfall_mm': 1400,
                'temperature_range_c': '(20, 40)'
            },
            {
                'location': 'Kolkata',
                'country': 'India',
                'latitude': 22.5726,
                'longitude': 88.3639,
                'air_quality_index': 90,
                'water_quality': 'Moderate',
                'vegetation_coverage': 0.20,
                'climate_zone': 'Tropical Wet and Dry',
                'annual_rainfall_mm': 1800,
                'temperature_range_c': '(12, 38)'
            }
        ]
        
        df = pd.DataFrame(env_data)
        df['env_id'] = range(1, len(df) + 1)
        
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def create_infrastructure_data(self) -> gpd.GeoDataFrame:
        """Create sample infrastructure data for demonstration."""
        infra_data = [
            # Global infrastructure data
            {
                'project_name': 'Brooklyn Bridge',
                'city': 'New York',
                'country': 'USA',
                'latitude': 40.7061,
                'longitude': -73.9969,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 1883,
                'description': 'Historic suspension bridge connecting Manhattan and Brooklyn',
                'capacity': 'High',
                'maintenance_status': 'Good'
            },
            {
                'project_name': 'London Underground',
                'city': 'London',
                'country': 'UK',
                'latitude': 51.5074,
                'longitude': -0.1278,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 1863,
                'description': 'World\'s first underground railway system',
                'capacity': 'Very High',
                'maintenance_status': 'Good'
            },
            {
                'project_name': 'Tokyo Skytree',
                'city': 'Tokyo',
                'country': 'Japan',
                'latitude': 35.7100,
                'longitude': 139.8107,
                'infrastructure_type': 'Communication',
                'status': 'Operational',
                'completion_year': 2012,
                'description': 'Tallest tower in Japan for broadcasting and observation',
                'capacity': 'High',
                'maintenance_status': 'Excellent'
            },
            
            # Indian infrastructure data
            {
                'project_name': 'Mumbai Metro',
                'city': 'Mumbai',
                'country': 'India',
                'latitude': 19.0760,
                'longitude': 72.8777,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 2014,
                'description': 'Rapid transit system serving Mumbai metropolitan region',
                'capacity': 'High',
                'maintenance_status': 'Good'
            },
            {
                'project_name': 'Delhi Metro',
                'city': 'Delhi',
                'country': 'India',
                'latitude': 28.7041,
                'longitude': 77.1025,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 2002,
                'description': 'Metro rail system serving Delhi NCR region',
                'capacity': 'Very High',
                'maintenance_status': 'Good'
            },
            {
                'project_name': 'Bangalore International Airport',
                'city': 'Bangalore',
                'country': 'India',
                'latitude': 13.1986,
                'longitude': 77.7066,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 2008,
                'description': 'Modern international airport with advanced facilities',
                'capacity': 'High',
                'maintenance_status': 'Excellent'
            },
            {
                'project_name': 'Chennai Port',
                'city': 'Chennai',
                'country': 'India',
                'latitude': 13.0827,
                'longitude': 80.2707,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 1881,
                'description': 'Major seaport handling container and bulk cargo',
                'capacity': 'High',
                'maintenance_status': 'Good'
            },
            {
                'project_name': 'Kolkata Metro',
                'city': 'Kolkata',
                'country': 'India',
                'latitude': 22.5726,
                'longitude': 88.3639,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 1984,
                'description': 'First metro system in India',
                'capacity': 'Medium',
                'maintenance_status': 'Moderate'
            },
            {
                'project_name': 'Pune Smart City',
                'city': 'Pune',
                'country': 'India',
                'latitude': 18.5204,
                'longitude': 73.8567,
                'infrastructure_type': 'Smart City',
                'status': 'Under Development',
                'completion_year': 2025,
                'description': 'Smart city initiative with digital infrastructure',
                'capacity': 'Medium',
                'maintenance_status': 'N/A'
            },
            {
                'project_name': 'Ahmedabad BRTS',
                'city': 'Ahmedabad',
                'country': 'India',
                'latitude': 23.0225,
                'longitude': 72.5714,
                'infrastructure_type': 'Transportation',
                'status': 'Operational',
                'completion_year': 2009,
                'description': 'Bus Rapid Transit System with dedicated corridors',
                'capacity': 'Medium',
                'maintenance_status': 'Good'
            }
        ]
        
        df = pd.DataFrame(infra_data)
        df['infra_id'] = range(1, len(df) + 1)
        
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def fetch_satellite_imagery_info(self, lat: float, lon: float, 
                                   radius_km: float = 10) -> Dict[str, Any]:
        """Fetch information about satellite imagery for a location."""
        # This is a mock implementation - in a real system, you'd integrate with
        # actual satellite imagery APIs like Google Earth Engine, Sentinel Hub, etc.
        
        return {
            'location': {'lat': lat, 'lon': lon},
            'coverage_radius_km': radius_km,
            'available_datasets': [
                'Sentinel-2 (10m resolution)',
                'Landsat 8/9 (30m resolution)',
                'MODIS (250m resolution)'
            ],
            'latest_image_date': '2024-01-15',
            'cloud_cover_percent': 15,
            'vegetation_index': 0.65,
            'urban_area_percent': 45,
            'water_bodies_detected': 2,
            'description': f'Satellite imagery available for area around {lat:.4f}, {lon:.4f}'
        }
    
    def process_satellite_data(self, imagery_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze satellite imagery data."""
        # Mock satellite data processing
        processed_data = {
            'land_use_analysis': {
                'urban_area': imagery_data.get('urban_area_percent', 0),
                'vegetation_coverage': imagery_data.get('vegetation_index', 0),
                'water_coverage': imagery_data.get('water_bodies_detected', 0) * 5,  # Estimate
                'bare_soil': 100 - (imagery_data.get('urban_area_percent', 0) + 
                                   imagery_data.get('vegetation_index', 0) * 100 + 
                                   imagery_data.get('water_bodies_detected', 0) * 5)
            },
            'environmental_indicators': {
                'air_quality_estimate': 'Good' if imagery_data.get('cloud_cover_percent', 0) < 20 else 'Moderate',
                'vegetation_health': 'Healthy' if imagery_data.get('vegetation_index', 0) > 0.5 else 'Moderate',
                'urban_heat_island_risk': 'Low' if imagery_data.get('vegetation_index', 0) > 0.6 else 'Medium'
            },
            'recommendations': [
                'Monitor vegetation changes over time',
                'Assess urban development impact',
                'Track water body variations'
            ]
        }
        
        return processed_data
    
    def create_satellite_imagery_data(self) -> gpd.GeoDataFrame:
        """Create comprehensive sample satellite imagery data for demonstration."""
        satellite_data = [
            # Global satellite imagery data
            {
                'location': 'New York',
                'country': 'USA',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'sensor': 'Sentinel-2',
                'resolution_m': 10,
                'acquisition_date': '2024-01-15',
                'cloud_cover_percent': 12,
                'vegetation_index_ndvi': 0.45,
                'urban_area_percent': 78,
                'water_bodies_count': 3,
                'image_quality': 'Excellent',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2'],
                'analysis_features': ['Urban development', 'Central Park vegetation', 'Hudson River'],
                'change_detection': 'Urban area increased by 2.3% since 2023',
                'environmental_impact': 'Heat island effect moderate, vegetation stable'
            },
            {
                'location': 'London',
                'country': 'UK',
                'latitude': 51.5074,
                'longitude': -0.1278,
                'sensor': 'Landsat-9',
                'resolution_m': 30,
                'acquisition_date': '2024-01-12',
                'cloud_cover_percent': 25,
                'vegetation_index_ndvi': 0.38,
                'urban_area_percent': 65,
                'water_bodies_count': 2,
                'image_quality': 'Good',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'Thermal'],
                'analysis_features': ['Thames River', 'Hyde Park', 'Urban sprawl'],
                'change_detection': 'Green spaces reduced by 1.1% in last year',
                'environmental_impact': 'Moderate air quality, good water management'
            },
            {
                'location': 'Tokyo',
                'country': 'Japan',
                'latitude': 35.6762,
                'longitude': 139.6503,
                'sensor': 'PlanetScope',
                'resolution_m': 3,
                'acquisition_date': '2024-01-18',
                'cloud_cover_percent': 8,
                'vegetation_index_ndvi': 0.52,
                'urban_area_percent': 82,
                'water_bodies_count': 4,
                'image_quality': 'Excellent',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR'],
                'analysis_features': ['Tokyo Bay', 'Imperial Palace gardens', 'Urban density'],
                'change_detection': 'New construction detected in 15 locations',
                'environmental_impact': 'High urban density, good green space preservation'
            },
            
            # Indian satellite imagery data
            {
                'location': 'Mumbai',
                'country': 'India',
                'latitude': 19.0760,
                'longitude': 72.8777,
                'sensor': 'Sentinel-2',
                'resolution_m': 10,
                'acquisition_date': '2024-01-16',
                'cloud_cover_percent': 18,
                'vegetation_index_ndvi': 0.32,
                'urban_area_percent': 85,
                'water_bodies_count': 5,
                'image_quality': 'Good',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2'],
                'analysis_features': ['Arabian Sea coastline', 'Sanjay Gandhi National Park', 'Mithi River'],
                'change_detection': 'Coastal erosion detected, mangrove loss 3.2%',
                'environmental_impact': 'High pollution levels, coastal ecosystem stress'
            },
            {
                'location': 'Delhi',
                'country': 'India',
                'latitude': 28.7041,
                'longitude': 77.1025,
                'sensor': 'Landsat-8',
                'resolution_m': 30,
                'acquisition_date': '2024-01-14',
                'cloud_cover_percent': 22,
                'vegetation_index_ndvi': 0.28,
                'urban_area_percent': 88,
                'water_bodies_count': 3,
                'image_quality': 'Moderate',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'Thermal'],
                'analysis_features': ['Yamuna River', 'Lodhi Gardens', 'Urban heat island'],
                'change_detection': 'Urban expansion 4.1%, green cover reduced 2.8%',
                'environmental_impact': 'Severe air pollution, water quality degradation'
            },
            {
                'location': 'Bangalore',
                'country': 'India',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'sensor': 'Sentinel-2',
                'resolution_m': 10,
                'acquisition_date': '2024-01-17',
                'cloud_cover_percent': 15,
                'vegetation_index_ndvi': 0.48,
                'urban_area_percent': 72,
                'water_bodies_count': 8,
                'image_quality': 'Excellent',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2'],
                'analysis_features': ['Lalbagh Botanical Garden', 'Bellandur Lake', 'Tech corridors'],
                'change_detection': 'Lake area reduced by 12%, tech parks expanded',
                'environmental_impact': 'Good green cover, lake pollution concerns'
            },
            {
                'location': 'Chennai',
                'country': 'India',
                'latitude': 13.0827,
                'longitude': 80.2707,
                'sensor': 'PlanetScope',
                'resolution_m': 3,
                'acquisition_date': '2024-01-13',
                'cloud_cover_percent': 20,
                'vegetation_index_ndvi': 0.35,
                'urban_area_percent': 75,
                'water_bodies_count': 6,
                'image_quality': 'Good',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR'],
                'analysis_features': ['Bay of Bengal', 'Adyar River', 'Beach erosion'],
                'change_detection': 'Coastal erosion 5.2%, new port development',
                'environmental_impact': 'Coastal ecosystem stress, urban flooding risk'
            },
            {
                'location': 'Kolkata',
                'country': 'India',
                'latitude': 22.5726,
                'longitude': 88.3639,
                'sensor': 'Landsat-9',
                'resolution_m': 30,
                'acquisition_date': '2024-01-19',
                'cloud_cover_percent': 28,
                'vegetation_index_ndvi': 0.42,
                'urban_area_percent': 80,
                'water_bodies_count': 12,
                'image_quality': 'Moderate',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'Thermal'],
                'analysis_features': ['Hooghly River', 'East Kolkata Wetlands', 'Sundarbans proximity'],
                'change_detection': 'Wetland area reduced 8.3%, river pollution increased',
                'environmental_impact': 'Critical wetland loss, biodiversity threat'
            },
            {
                'location': 'Hyderabad',
                'country': 'India',
                'latitude': 17.3850,
                'longitude': 78.4867,
                'sensor': 'Sentinel-2',
                'resolution_m': 10,
                'acquisition_date': '2024-01-20',
                'cloud_cover_percent': 16,
                'vegetation_index_ndvi': 0.38,
                'urban_area_percent': 70,
                'water_bodies_count': 4,
                'image_quality': 'Excellent',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2'],
                'analysis_features': ['Hussain Sagar Lake', 'IT corridors', 'Rock formations'],
                'change_detection': 'IT corridor expansion 6.7%, lake water quality stable',
                'environmental_impact': 'Balanced development, good water management'
            },
            {
                'location': 'Pune',
                'country': 'India',
                'latitude': 18.5204,
                'longitude': 73.8567,
                'sensor': 'Landsat-8',
                'resolution_m': 30,
                'acquisition_date': '2024-01-21',
                'cloud_cover_percent': 19,
                'vegetation_index_ndvi': 0.45,
                'urban_area_percent': 68,
                'water_bodies_count': 5,
                'image_quality': 'Good',
                'bands_available': ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2', 'Thermal'],
                'analysis_features': ['Mula-Mutha Rivers', 'Sinhagad Fort', 'Educational institutions'],
                'change_detection': 'Educational hub expansion, river pollution moderate',
                'environmental_impact': 'Good green cover, moderate water quality'
            }
        ]
        
        df = pd.DataFrame(satellite_data)
        df['satellite_id'] = range(1, len(df) + 1)
        
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def analyze_satellite_imagery(self, location: str, coordinates: Tuple[float, float], 
                                 sensor_type: str = 'Sentinel-2') -> Dict[str, Any]:
        """Analyze satellite imagery for a specific location."""
        lat, lon = coordinates
        
        # Simulate satellite imagery analysis
        analysis_results = {
            'location': location,
            'coordinates': {'latitude': lat, 'longitude': lon},
            'sensor_used': sensor_type,
            'analysis_timestamp': datetime.now().isoformat(),
            'image_characteristics': {
                'resolution': self._get_sensor_resolution(sensor_type),
                'coverage_area_km2': random.uniform(100, 1000),
                'temporal_coverage': '2020-2024',
                'spectral_bands': self._get_sensor_bands(sensor_type)
            },
            'land_use_analysis': {
                'urban_area_percent': random.uniform(20, 90),
                'vegetation_coverage_percent': random.uniform(10, 60),
                'water_bodies_percent': random.uniform(2, 15),
                'bare_soil_percent': random.uniform(5, 30),
                'agricultural_land_percent': random.uniform(0, 40)
            },
            'environmental_indicators': {
                'vegetation_health_index': random.uniform(0.2, 0.8),
                'urban_heat_island_intensity': random.uniform(0.5, 3.0),
                'air_quality_estimate': random.choice(['Good', 'Moderate', 'Poor', 'Very Poor']),
                'water_quality_estimate': random.choice(['Excellent', 'Good', 'Moderate', 'Poor']),
                'biodiversity_index': random.uniform(0.3, 0.9)
            },
            'change_detection': {
                'urban_expansion_rate': f"{random.uniform(1.0, 5.0):.1f}% per year",
                'vegetation_change': f"{random.uniform(-3.0, 2.0):.1f}% per year",
                'water_body_changes': f"{random.uniform(-2.0, 1.0):.1f}% per year",
                'land_use_conversion': random.choice(['Low', 'Moderate', 'High'])
            },
            'spatial_analysis': {
                'nearest_water_body_km': random.uniform(0.5, 10.0),
                'elevation_range_m': f"{random.uniform(0, 100):.0f}-{random.uniform(100, 500):.0f}",
                'slope_analysis': random.choice(['Flat', 'Gentle', 'Moderate', 'Steep']),
                'aspect_direction': random.choice(['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest'])
            },
            'recommendations': [
                'Monitor vegetation changes monthly',
                'Assess urban development impact quarterly',
                'Track water body variations seasonally',
                'Evaluate environmental quality annually',
                'Implement sustainable development practices'
            ],
            'data_quality_metrics': {
                'cloud_cover_percent': random.uniform(5, 30),
                'atmospheric_correction': 'Applied',
                'geometric_accuracy_m': random.uniform(1, 10),
                'radiometric_calibration': 'Calibrated',
                'temporal_consistency': random.choice(['High', 'Medium', 'Low'])
            }
        }
        
        return analysis_results
    
    def _get_sensor_resolution(self, sensor: str) -> str:
        """Get resolution for different satellite sensors."""
        resolutions = {
            'Sentinel-2': '10m (multispectral), 20m (SWIR), 60m (atmospheric)',
            'Landsat-8': '30m (multispectral), 15m (panchromatic), 100m (thermal)',
            'Landsat-9': '30m (multispectral), 15m (panchromatic), 100m (thermal)',
            'PlanetScope': '3-5m (multispectral)',
            'MODIS': '250m-1km (various bands)',
            'RapidEye': '5m (multispectral)',
            'SPOT': '2.5m-20m (various)',
            'IKONOS': '1m (panchromatic), 4m (multispectral)',
            'QuickBird': '0.6m (panchromatic), 2.4m (multispectral)',
            'WorldView': '0.3m (panchromatic), 1.2m (multispectral)'
        }
        return resolutions.get(sensor, 'Unknown')
    
    def _get_sensor_bands(self, sensor: str) -> List[str]:
        """Get available spectral bands for different sensors."""
        bands = {
            'Sentinel-2': ['Blue', 'Green', 'Red', 'Red Edge 1-3', 'NIR', 'SWIR1', 'SWIR2'],
            'Landsat-8': ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Panchromatic', 'Thermal 1-2'],
            'Landsat-9': ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Panchromatic', 'Thermal 1-2'],
            'PlanetScope': ['Blue', 'Green', 'Red', 'NIR'],
            'MODIS': ['Red', 'NIR', 'Blue', 'Green', 'SWIR', 'Thermal'],
            'RapidEye': ['Blue', 'Green', 'Red', 'Red Edge', 'NIR'],
            'SPOT': ['Green', 'Red', 'NIR', 'SWIR'],
            'IKONOS': ['Blue', 'Green', 'Red', 'NIR', 'Panchromatic'],
            'QuickBird': ['Blue', 'Green', 'Red', 'NIR', 'Panchromatic'],
            'WorldView': ['Blue', 'Green', 'Red', 'NIR', 'Red Edge', 'Yellow', 'Panchromatic']
        }
        return bands.get(sensor, ['Unknown'])
    
    def generate_satellite_time_series(self, location: str, coordinates: Tuple[float, float], 
                                     start_date: str, end_date: str, 
                                     sensor: str = 'Sentinel-2') -> pd.DataFrame:
        """Generate time series data for satellite imagery analysis."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        ndvi_values = []
        urban_percent = []
        cloud_cover = []
        
        current_date = start
        while current_date <= end:
            dates.append(current_date.strftime('%Y-%m-%d'))
            # Simulate seasonal variations
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
            ndvi_values.append(max(0.1, min(0.9, random.uniform(0.3, 0.7) * seasonal_factor)))
            urban_percent.append(min(95, max(20, random.uniform(60, 80) + random.uniform(-5, 5))))
            cloud_cover.append(max(0, min(50, random.uniform(10, 30) + random.uniform(-10, 10))))
            current_date += timedelta(days=30)  # Monthly data
        
        time_series_data = {
            'date': dates,
            'ndvi': ndvi_values,
            'urban_area_percent': urban_percent,
            'cloud_cover_percent': cloud_cover,
            'location': location,
            'sensor': sensor,
            'latitude': coordinates[0],
            'longitude': coordinates[1]
        }
        
        return pd.DataFrame(time_series_data)
    
    def detect_environmental_changes(self, time_series_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect environmental changes from satellite time series data."""
        if len(time_series_df) < 2:
            return {'error': 'Insufficient data for change detection'}
        
        # Calculate trends
        ndvi_trend = np.polyfit(range(len(time_series_df)), time_series_df['ndvi'], 1)[0]
        urban_trend = np.polyfit(range(len(time_series_df)), time_series_df['urban_area_percent'], 1)[0]
        
        # Detect anomalies
        ndvi_mean = time_series_df['ndvi'].mean()
        ndvi_std = time_series_df['ndvi'].std()
        ndvi_anomalies = time_series_df[abs(time_series_df['ndvi'] - ndvi_mean) > 2 * ndvi_std]
        
        change_analysis = {
            'trends': {
                'ndvi_trend': f"{ndvi_trend:.4f} per month",
                'urban_expansion_trend': f"{urban_trend:.2f}% per month",
                'ndvi_direction': 'Increasing' if ndvi_trend > 0 else 'Decreasing',
                'urban_direction': 'Expanding' if urban_trend > 0 else 'Contracting'
            },
            'anomalies': {
                'ndvi_anomaly_count': len(ndvi_anomalies),
                'anomaly_dates': ndvi_anomalies['date'].tolist() if len(ndvi_anomalies) > 0 else [],
                'anomaly_severity': 'High' if len(ndvi_anomalies) > len(time_series_df) * 0.2 else 'Low'
            },
            'seasonal_patterns': {
                'ndvi_variability': time_series_df['ndvi'].std() / time_series_df['ndvi'].mean(),
                'urban_stability': time_series_df['urban_area_percent'].std() / time_series_df['urban_area_percent'].mean(),
                'seasonal_strength': 'Strong' if time_series_df['ndvi'].std() > 0.1 else 'Weak'
            },
            'environmental_health': {
                'vegetation_health': 'Good' if ndvi_trend >= 0 else 'Declining',
                'urban_impact': 'Moderate' if abs(urban_trend) < 1 else 'High',
                'overall_status': 'Stable' if abs(ndvi_trend) < 0.01 else 'Changing'
            }
        }
        
        return change_analysis
    
    def create_satellite_analysis_report(self, analysis_data: Dict[str, Any], 
                                       time_series_data: Optional[pd.DataFrame] = None) -> str:
        """Create a comprehensive satellite analysis report."""
        report = f"""
# Satellite Imagery Analysis Report

## Location Information
- **Location**: {analysis_data['location']}
- **Coordinates**: {analysis_data['coordinates']['latitude']:.4f}, {analysis_data['coordinates']['longitude']:.4f}
- **Sensor**: {analysis_data['sensor_used']}
- **Analysis Date**: {analysis_data['analysis_timestamp'][:10]}

## Land Use Analysis
- **Urban Area**: {analysis_data['land_use_analysis']['urban_area_percent']:.1f}%
- **Vegetation Coverage**: {analysis_data['land_use_analysis']['vegetation_coverage_percent']:.1f}%
- **Water Bodies**: {analysis_data['land_use_analysis']['water_bodies_percent']:.1f}%
- **Agricultural Land**: {analysis_data['land_use_analysis']['agricultural_land_percent']:.1f}%

## Environmental Indicators
- **Vegetation Health Index**: {analysis_data['environmental_indicators']['vegetation_health_index']:.3f}
- **Urban Heat Island Intensity**: {analysis_data['environmental_indicators']['urban_heat_island_intensity']:.1f}°C
- **Air Quality Estimate**: {analysis_data['environmental_indicators']['air_quality_estimate']}
- **Biodiversity Index**: {analysis_data['environmental_indicators']['biodiversity_index']:.3f}

## Change Detection
- **Urban Expansion Rate**: {analysis_data['change_detection']['urban_expansion_rate']}
- **Vegetation Change**: {analysis_data['change_detection']['vegetation_change']}
- **Land Use Conversion**: {analysis_data['change_detection']['land_use_conversion']}

## Spatial Analysis
- **Nearest Water Body**: {analysis_data['spatial_analysis']['nearest_water_body_km']:.1f} km
- **Elevation Range**: {analysis_data['spatial_analysis']['elevation_range_m']} m
- **Slope**: {analysis_data['spatial_analysis']['slope_analysis']}
- **Aspect**: {analysis_data['spatial_analysis']['aspect_direction']}

## Data Quality
- **Cloud Cover**: {analysis_data['data_quality_metrics']['cloud_cover_percent']:.1f}%
- **Geometric Accuracy**: ±{analysis_data['data_quality_metrics']['geometric_accuracy_m']:.1f} m
- **Temporal Consistency**: {analysis_data['data_quality_metrics']['temporal_consistency']}

## Recommendations
"""
        
        for rec in analysis_data['recommendations']:
            report += f"- {rec}\n"
        
        if time_series_data is not None:
            change_analysis = self.detect_environmental_changes(time_series_data)
            report += f"""
## Time Series Analysis
- **NDVI Trend**: {change_analysis['trends']['ndvi_trend']}
- **Urban Expansion Trend**: {change_analysis['trends']['urban_expansion_trend']}
- **Environmental Health**: {change_analysis['environmental_health']['overall_status']}
- **Anomaly Detection**: {change_analysis['anomalies']['ndvi_anomaly_count']} anomalies detected
"""
        
        return report.strip()
    
    def export_data(self, gdf: gpd.GeoDataFrame, file_path: str, 
                   format_type: str = 'geojson') -> bool:
        """Export geographic data to various formats."""
        try:
            if format_type == 'geojson':
                gdf.to_file(file_path, driver='GeoJSON')
            elif format_type == 'csv':
                # Export without geometry column
                df_export = gdf.drop(columns=['geometry'])
                df_export.to_csv(file_path, index=False)
            elif format_type == 'shp':
                gdf.to_file(file_path, driver='ESRI Shapefile')
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Data exported successfully to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False
    
    def validate_geographic_data(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Validate geographic data quality and completeness."""
        validation_results = {
            'total_records': len(gdf),
            'valid_geometry': gdf.geometry.is_valid.sum(),
            'invalid_geometry': (~gdf.geometry.is_valid).sum(),
            'null_geometry': gdf.geometry.isna().sum(),
            'coordinate_range': {
                'min_lat': gdf.geometry.bounds.miny.min(),
                'max_lat': gdf.geometry.bounds.maxy.max(),
                'min_lon': gdf.geometry.bounds.minx.min(),
                'max_lon': gdf.geometry.bounds.maxx.max()
            },
            'crs_info': str(gdf.crs),
            'data_quality_score': 0.0
        }
        
        # Calculate quality score
        total_checks = 4
        passed_checks = 0
        
        if validation_results['valid_geometry'] == validation_results['total_records']:
            passed_checks += 1
        if validation_results['null_geometry'] == 0:
            passed_checks += 1
        if gdf.crs is not None:
            passed_checks += 1
        if validation_results['total_records'] > 0:
            passed_checks += 1
        
        validation_results['data_quality_score'] = passed_checks / total_checks
        
        return validation_results

# Global instance
data_utils = DataUtils()
