import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static, st_folium
import logging
import re

from config import config
from rag.rag_system import GeographicRAGSystem
from evaluation.rag_evaluator import RAGEvaluator
from utils.data_utils import data_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Geographic Information RAG System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_system():
    try:
        config.validate()
        rag_system = GeographicRAGSystem()
        evaluator = RAGEvaluator()

        counts = rag_system.get_collection_counts()
        needs_ingest = (
            counts.get('geographic_data', 0) == 0 or
            counts.get('environmental_data', 0) == 0 or
            counts.get('infrastructure_data', 0) == 0 or
            counts.get('satellite_data', 0) == 0
        )
        if needs_ingest:
            cities_df = data_utils.create_sample_geographic_data()
            env_gdf = data_utils.create_environmental_data()
            infra_gdf = data_utils.create_infrastructure_data()
            satellite_gdf = data_utils.create_satellite_imagery_data()
            rag_system.ingest_sample_data(cities_df, env_gdf, infra_gdf, satellite_gdf)
        return rag_system, evaluator
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None

def infer_collection_from_query(query: str) -> str:
    text = (query or "").lower()
    env_kw = [
        r"\bair quality\b", r"\baqi\b", r"\bwater quality\b", r"\brainfall\b", r"\bclimate\b",
        r"\btemperature\b", r"\bvegetation\b", r"\bpollution\b", r"\benvironment\b", r"\bhumidity\b"
    ]
    infra_kw = [
        r"\bmetro\b", r"\bairport\b", r"\bport\b", r"\bbridge\b", r"\broad\b", r"\brail\b",
        r"\binfrastructure\b", r"\bbrts\b", r"\bsmart city\b", r"\bstation\b", r"\bhighway\b"
    ]
    satellite_kw = [
        r"\bsatellite\b", r"\bimagery\b", r"\bndvi\b", r"\bremote sensing\b", r"\blandsat\b",
        r"\bsentinel\b", r"\bvegetation index\b", r"\bland use\b", r"\bchange detection\b",
        r"\burban expansion\b", r"\bcoastal erosion\b", r"\bdeforestation\b", r"\bheat island\b"
    ]
    if any(re.search(p, text) for p in satellite_kw):
        return "satellite_data"
    elif any(re.search(p, text) for p in env_kw):
        return "environmental_data"
    elif any(re.search(p, text) for p in infra_kw):
        return "infrastructure_data"
    else:
        return "geographic_data"  # Default

def main():
    st.title("🌍 Geographic Information RAG System")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Choose a page:")

    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "🏠 Home"
    if st.sidebar.button("🔍 Query System", use_container_width=True):
        st.session_state.current_page = "🔍 Query System"
    if st.sidebar.button("🗺️ Maps", use_container_width=True):
        st.session_state.current_page = "🗺️ Maps"
    if st.sidebar.button("📊 Analysis", use_container_width=True):
        st.session_state.current_page = "📊 Analysis"
    if st.sidebar.button("🛰️ Satellite Imagery", use_container_width=True):
        st.session_state.current_page = "🛰️ Satellite Imagery"
    if st.sidebar.button("📈 Evaluation", use_container_width=True):
        st.session_state.current_page = "📈 Evaluation"

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system, st.session_state.evaluator = initialize_system()

    if st.session_state.rag_system is None:
        st.error("System not initialized. Please check your configuration.")
        return

    if 'sample_data' not in st.session_state:
        cities, env, infra, satellite = (
            data_utils.create_sample_geographic_data(),
            data_utils.create_environmental_data(),
            data_utils.create_infrastructure_data(),
            data_utils.create_satellite_imagery_data(),
        )
        if cities is not None:
            st.session_state.sample_data = {
                'cities': cities, 'environmental': env, 'infrastructure': infra, 'satellite': satellite
            }

    if st.session_state.current_page == "🏠 Home":
        show_home_page()
    elif st.session_state.current_page == "🔍 Query System":
        show_query_page()
    elif st.session_state.current_page == "🗺️ Maps":
        show_maps_page()
    elif st.session_state.current_page == "📊 Analysis":
        show_analysis_page()
    elif st.session_state.current_page == "🛰️ Satellite Imagery":
        show_satellite_imagery_page()
    elif st.session_state.current_page == "📈 Evaluation":
        show_evaluation_page()

def show_home_page():
    st.header("Welcome to the Geographic Information RAG System")
    st.markdown(
        """
        - Intelligent spatial queries
        - Satellite imagery analysis and interpretation
        - Multi-scale geographic analysis
        - Interactive maps and charts
        - DeepEval performance metrics
        """
    )

    st.subheader("🇮🇳 India-Focused Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            - Major cities: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata
            - Also: Pune, Ahmedabad, Jaipur, Surat
            - Environmental: AQI, water, climate, vegetation
            """
        )
    with col2:
        st.markdown(
            """
            - Infrastructure: Metro (Delhi, Mumbai, Kolkata)
            - Smart city initiatives
            - Ports (Chennai), Airports (Bangalore)
            - Satellite monitoring: Urban expansion, coastal erosion
            """
        )

    if st.button("Check System Status"):
        with st.spinner("Checking system status..."):
            status = st.session_state.rag_system.get_system_status()
            if status.get('system_status') == 'Operational':
                st.success("✅ System is operational")
                st.json(status)
            else:
                st.error(f"❌ System error: {status.get('error', 'Unknown error')}")

def show_query_page():
    st.header("🔍 Geographic Query System")
    with st.expander("What is 'geographic_data'?"):
        st.markdown("City profiles with coordinates used for spatial queries and semantic search.")

    st.markdown(
        """
        Examples: "What is the air quality in Delhi?", "Tell me about Mumbai Metro",
        "How is the climate in Bangalore?", "Ports in Chennai?",
        "Show me satellite imagery of Delhi", "What is the NDVI in Mumbai?",
        "Analyze land use changes in Bangalore", "Detect urban expansion in Chennai"
        """
    )

    user_query = st.text_area("Enter your geographic query:", height=100,
                              placeholder="e.g., What is the air quality in Delhi?")

    use_coordinates = st.checkbox("Use specific coordinates")
    coordinates = None
    radius_km = None
    lat, lon = 28.7041, 77.1025
    if use_coordinates:
        st.caption("Click on the map to pick coordinates. Default center: Delhi")
        lat = st.number_input("Latitude", value=lat)
        lon = st.number_input("Longitude", value=lon)
        radius_km = st.slider("Search radius (km)", 1, 100, 10)
        m = folium.Map(location=[lat, lon], zoom_start=6)
        folium.Marker([lat, lon], tooltip="Current").add_to(m)
        map_state = st_folium(m, height=350, width=700)
        if map_state and map_state.get("last_clicked"):
            lat = map_state["last_clicked"]["lat"]
            lon = map_state["last_clicked"]["lng"]
            st.info(f"Selected coordinates: {lat:.4f}, {lon:.4f}")
        coordinates = {'latitude': lat, 'longitude': lon}

    inferred_collection = infer_collection_from_query(user_query)
    st.caption(f"Detected data collection: {inferred_collection}")

    if st.button("🚀 Execute Query"):
        if user_query.strip():
            with st.spinner("Processing your query..."):
                try:
                    result = st.session_state.rag_system.query(
                        user_query, coordinates, radius_km, inferred_collection
                    )
                    st.subheader("📋 Query Results")
                    st.markdown("**AI Response:**")
                    st.write(result['response'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("Documents Retrieved", len(result['retrieved_documents']))
                    with col3:
                        st.metric("Collection", inferred_collection)
                    if result['retrieved_documents']:
                        st.subheader("📚 Retrieved Documents")
                        for i, doc in enumerate(result['retrieved_documents']):
                            with st.expander(f"Document {i+1}"):
                                st.json(doc)
                    if result['context_summary']:
                        st.subheader("🔍 Context Summary")
                        st.write(result['context_summary'])
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a query to proceed.")

def show_maps_page():
    st.header("🗺️ Interactive Maps")
    if 'sample_data' not in st.session_state:
        st.error("Sample data not loaded.")
        return

    data_type = st.selectbox("Select data to display:", ["Cities", "Environmental", "Infrastructure"])
    data = st.session_state.sample_data['cities'] if data_type == "Cities" else (
        st.session_state.sample_data['environmental'] if data_type == "Environmental" else st.session_state.sample_data['infrastructure']
    )

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    for _, row in data.iterrows():
        if 'geometry' in row and row.geometry and hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
            lat, lon = row.geometry.y, row.geometry.x
        elif 'latitude' in row and 'longitude' in row:
            lat, lon = row['latitude'], row['longitude']
        else:
            continue
        popup_title = row.get('name', row.get('location', 'Location'))
        popup_text = f"<b>{popup_title}</b><br>"
        for col in data.columns:
            if col != 'geometry' and pd.notna(row[col]):
                popup_text += f"{col}: {row[col]}<br>"
        folium.Marker([lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
    folium_static(m, width=800, height=600)

def show_analysis_page():
    st.header("📊 Data Analysis")
    if 'sample_data' not in st.session_state:
        st.error("Sample data not loaded.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cities", len(st.session_state.sample_data['cities']))
    with col2:
        st.metric("Environmental Stations", len(st.session_state.sample_data['environmental']))
    with col3:
        st.metric("Infrastructure Points", len(st.session_state.sample_data['infrastructure']))

    st.subheader("Cities Data")
    st.dataframe(st.session_state.sample_data['cities'].drop(columns=['geometry'], errors='ignore'), use_container_width=True)
    st.subheader("Environmental Data")
    st.dataframe(st.session_state.sample_data['environmental'].drop(columns=['geometry'], errors='ignore'), use_container_width=True)

def show_evaluation_page():
    st.header("📈 System Evaluation")
    st.write("Run system evaluation to measure performance metrics.")
    if st.button("🚀 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                test_cases = st.session_state.evaluator.create_test_cases()
                if not test_cases:
                    st.warning("DeepEval metrics are not available or no test cases could be created.")
                    return
                report = st.session_state.evaluator.run_comprehensive_evaluation(test_cases)
                if 'error' not in report:
                    st.success("✅ Evaluation completed!")
                    summary = st.session_state.evaluator.generate_performance_summary(report)
                    st.text_area("Performance Summary", summary, height=300)
                else:
                    st.error(f"Evaluation failed: {report['error']}")
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

def show_satellite_imagery_page():
    st.header("🛰️ Satellite Imagery Analysis")
    st.markdown("""
    Analyze satellite imagery data for environmental monitoring, land use analysis, and change detection.
    """)
    
    # Satellite data overview
    if 'sample_data' in st.session_state and 'satellite' in st.session_state.sample_data:
        st.subheader("📊 Satellite Data Overview")
        satellite_data = st.session_state.sample_data['satellite']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Locations", len(satellite_data))
        with col2:
            st.metric("Sensors", satellite_data['sensor'].nunique())
        with col3:
            st.metric("Avg Cloud Cover", f"{satellite_data['cloud_cover_percent'].mean():.1f}%")
        with col4:
            st.metric("Avg NDVI", f"{satellite_data['vegetation_index_ndvi'].mean():.3f}")
    
    # Interactive satellite analysis
    st.subheader("🔍 Interactive Satellite Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Location selection
        location_options = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
        selected_location = st.selectbox("Select Location:", location_options)
        
        # Get coordinates for selected location
        location_coords = {
            "Delhi": (28.7041, 77.1025),
            "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946),
            "Chennai": (13.0827, 80.2707),
            "Kolkata": (22.5726, 88.3639),
            "Hyderabad": (17.3850, 78.4867),
            "Pune": (18.5204, 73.8567)
        }
        
        coordinates = location_coords[selected_location]
        
        # Sensor selection
        sensor_options = ["Sentinel-2", "Landsat-8", "Landsat-9", "PlanetScope", "MODIS"]
        selected_sensor = st.selectbox("Select Sensor:", sensor_options, index=0)
        
        # Analysis parameters
        start_date = st.date_input("Start Date:", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date:", value=pd.to_datetime("2024-01-01"))
    
    with col2:
        st.subheader("📍 Location Info")
        st.write(f"**City:** {selected_location}")
        st.write(f"**Coordinates:** {coordinates[0]:.4f}, {coordinates[1]:.4f}")
        st.write(f"**Sensor:** {selected_sensor}")
        st.write(f"**Time Period:** {start_date} to {end_date}")
    
    # Run analysis
    if st.button("🚀 Analyze Satellite Imagery"):
        with st.spinner("Analyzing satellite imagery..."):
            try:
                # Perform satellite analysis
                analysis_results = data_utils.analyze_satellite_imagery(
                    selected_location, coordinates, selected_sensor
                )
                
                # Generate time series data
                time_series_data = data_utils.generate_satellite_time_series(
                    selected_location, coordinates, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d'), 
                    selected_sensor
                )
                
                # Create comprehensive report
                report = data_utils.create_satellite_analysis_report(
                    analysis_results, time_series_data
                )
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Show key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Urban Area", f"{analysis_results['land_use_analysis']['urban_area_percent']:.1f}%")
                with col2:
                    st.metric("Vegetation", f"{analysis_results['land_use_analysis']['vegetation_coverage_percent']:.1f}%")
                with col3:
                    st.metric("Water Bodies", f"{analysis_results['land_use_analysis']['water_bodies_percent']:.1f}%")
                with col4:
                    st.metric("NDVI", f"{analysis_results['environmental_indicators']['vegetation_health_index']:.3f}")
                
                # Display full report
                st.subheader("📋 Analysis Report")
                st.markdown(report)
                
                # Show time series data
                st.subheader("📈 Time Series Analysis")
                if not time_series_data.empty:
                    # NDVI trend
                    st.line_chart(time_series_data.set_index('date')['ndvi'])
                    st.caption("NDVI (Normalized Difference Vegetation Index) over time")
                    
                    # Urban area changes
                    st.line_chart(time_series_data.set_index('date')['urban_area_percent'])
                    st.caption("Urban area percentage over time")
                    
                    # Cloud cover
                    st.line_chart(time_series_data.set_index('date')['cloud_cover_percent'])
                    st.caption("Cloud cover percentage over time")
                
            except Exception as e:
                st.error(f"Error during satellite analysis: {str(e)}")
    
    # Satellite data table
    if 'sample_data' in st.session_state and 'satellite' in st.session_state.sample_data:
        st.subheader("🛰️ Satellite Data Table")
        satellite_data = st.session_state.sample_data['satellite']
        # Drop geometry column for display
        display_data = satellite_data.drop(columns=['geometry'], errors='ignore')
        st.dataframe(display_data, use_container_width=True)
    
    # Satellite imagery insights
    st.subheader("💡 Key Insights")
    st.markdown("""
    - **High-resolution imagery** (3-10m) for detailed urban analysis
    - **Multi-spectral bands** for vegetation health monitoring
    - **Temporal analysis** for change detection and trend analysis
    - **Environmental indicators** including NDVI, urban heat island effects
    - **Land use classification** and change detection capabilities
    """)

if __name__ == "__main__":
    main()
