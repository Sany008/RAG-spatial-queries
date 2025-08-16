import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
from typing import List, Tuple, Union, Optional
import pyproj
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static

class GeoUtils:
    """Utility class for geographic operations and spatial analysis."""
    
    def __init__(self, default_crs: str = "EPSG:4326"):
        self.default_crs = default_crs
        self.wgs84 = pyproj.CRS("EPSG:4326")
    
    def create_point(self, lat: float, lon: float, crs: str = None) -> Point:
        """Create a Point geometry from coordinates."""
        return Point(lon, lat)
    
    def create_bounding_box(self, center_lat: float, center_lon: float, 
                           radius_km: float) -> Polygon:
        """Create a bounding box around a center point."""
        # Convert radius to degrees (approximate)
        lat_degree = radius_km / 111.32
        lon_degree = radius_km / (111.32 * np.cos(np.radians(center_lat)))
        
        min_lat = center_lat - lat_degree
        max_lat = center_lat + lat_degree
        min_lon = center_lon - lon_degree
        max_lon = center_lon + lon_degree
        
        return box(min_lon, min_lat, max_lon, max_lat)
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate distance between two points in kilometers."""
        return geodesic(point1, point2).kilometers
    
    def is_point_in_polygon(self, point: Point, polygon: Polygon) -> bool:
        """Check if a point is within a polygon."""
        return polygon.contains(point)
    
    def buffer_point(self, point: Point, radius_km: float, 
                    crs: str = None) -> Polygon:
        """Create a buffer around a point."""
        if crs and crs != self.default_crs:
            # Transform to projected CRS for accurate buffering
            proj_crs = pyproj.CRS(crs)
            transformer = pyproj.Transformer.from_crs(
                self.wgs84, proj_crs, always_xy=True
            )
            point_proj = transform(transformer.transform, point)
            buffer_proj = point_proj.buffer(radius_km * 1000)  # Convert to meters
            transformer_back = pyproj.Transformer.from_crs(
                proj_crs, self.wgs84, always_xy=True
            )
            return transform(transformer_back.transform, buffer_proj)
        else:
            # Simple approximation for WGS84
            return point.buffer(radius_km / 111.32)
    
    def create_geodataframe(self, data: List[dict], 
                           geometry_column: str = "geometry") -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from a list of dictionaries with geometry."""
        return gpd.GeoDataFrame(data, crs=self.default_crs)
    
    def spatial_join(self, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                     how: str = "inner") -> gpd.GeoDataFrame:
        """Perform spatial join between two GeoDataFrames."""
        return gpd.sjoin(gdf1, gdf2, how=how, predicate="intersects")
    
    def create_interactive_map(self, center_lat: float = 40.7128, 
                              center_lon: float = -74.0060, 
                              zoom: int = 10) -> folium.Map:
        """Create an interactive Folium map."""
        return folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles="OpenStreetMap"
        )
    
    def add_markers_to_map(self, map_obj: folium.Map, 
                           locations: List[Tuple[float, float, str]]) -> folium.Map:
        """Add markers to a Folium map."""
        for lat, lon, popup in locations:
            folium.Marker(
                [lat, lon],
                popup=popup,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(map_obj)
        return map_obj
    
    def add_polygon_to_map(self, map_obj: folium.Map, 
                           polygon: Polygon, 
                           color: str = "blue", 
                           weight: int = 2) -> folium.Map:
        """Add a polygon to a Folium map."""
        folium.Polygon(
            locations=list(polygon.exterior.coords),
            color=color,
            weight=weight,
            fill=True,
            fillOpacity=0.2
        ).add_to(map_obj)
        return map_obj
    
    def coordinate_transform(self, point: Point, 
                           from_crs: str, 
                           to_crs: str) -> Point:
        """Transform coordinates between different CRS."""
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        return transform(transformer.transform, point)
    
    def calculate_centroid(self, geometry: Union[Point, Polygon]) -> Point:
        """Calculate the centroid of a geometry."""
        return geometry.centroid
    
    def simplify_geometry(self, geometry: Union[Point, Polygon], 
                         tolerance: float) -> Union[Point, Polygon]:
        """Simplify geometry while preserving topology."""
        return geometry.simplify(tolerance, preserve_topology=True)

# Global instance
geo_utils = GeoUtils()
