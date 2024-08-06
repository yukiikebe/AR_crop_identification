import streamlit as st
from streamlit_folium import st_folium
import folium
import json
from geopy.geocoders import Nominatim
import ee
import geemap
from PIL import Image
import os
from datetime import datetime, timedelta
from folium.plugins import Draw
import shutil 
from data.Arkansas.Download import download_dataset
from data.Arkansas.pre_processing import preprocess_AR
from train_and_eval.inference import inference_AR
ee.Authenticate()
ee.Initialize(project='ee-vvuonghn')

MAPBOX_TOKEN = "pk.eyJ1IjoidnVvbmdoIiwiYSI6ImNseWRobHVtZTA0a2EyaW9wc2loM2cxOWIifQ.U5utf4cO8ldi087Mn-h0FA"

config = {"save_satellite_dir": '/home/vuonghn/research/dataset/satellite/arkansas/satellite_images/2023/',
          "cdl_dir": "/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/",
          "processed_dir": "/home/vuonghn/research/dataset/satellite/arkansas/preprocessed_data/preprocessed_demo",
          "output_vis": 'output/visualized_rgb_with_legend.png',
          "model_config": 'configs/Arkansas/TSViT_inference.yaml'}

def get_map():
    st.markdown("#### Selection the ROI")
    # Initialize a Folium map using Mapbox tiles
    
    m = folium.Map(
        location=[36.0688455727019, -94.17536208601327],
        zoom_start=10,
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        # tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        attr='Mapbox attribution'
    )

    # Add drawing options to the map
    draw = Draw(export=True)
    draw.add_to(m)

    # Sidebar for location search
    st.sidebar.title("Location Search")
    query = st.sidebar.text_input("Enter a location (city, state, country):")
    # get_start_end_date()
    # Initialize the geolocator
    geolocator = Nominatim(user_agent="geoapp")

    # Default location (Central USA)
    default_location = [36.0688455727019, -94.17536208601327]
    zoom_start = 10

    if query:
        # Attempt to geocode the query
        location = geolocator.geocode(query)
        if location:
            map_location = [location.latitude, location.longitude]
            zoom_start = 12  # Zoom in closer for searched locations
        else:
            st.sidebar.error("Location not found. Showing default location.")
            map_location = default_location
    else:
        map_location = default_location

    # Update map view based on location search
    m.location = map_location
    m.zoom_start = zoom_start
    
    # Display the map with Streamlit
    map_data = st_folium(m)
    return map_data

def get_start_end_date():
    # Title for your app
    st.sidebar.title('Date Range')

    # Create a start date input widget
    start_date = st.sidebar.date_input("Select a start date:", datetime.today())
    
    # Create an end date input widget
    # By default, set it to one week from the start date
    end_date = st.sidebar.date_input("Select an end date:", start_date + timedelta(days=60))
    vis_day = st.sidebar.date_input("Visualization date:", start_date + timedelta(days=30))

    # Check if the start date is after the end date and display a warning
    if start_date > end_date or vis_day > end_date or vis_day < start_date:
        st.sidebar.error('Error: End date must be after start date.')

    # Display the selected date range
    st.sidebar.write('You selected:', start_date, 'to', end_date)
    return str(start_date), str(end_date), str(vis_day)

def get_roi(map_data):

    if map_data and isinstance(map_data, dict) and 'all_drawings' in map_data and map_data['all_drawings'] is not None:
        # st.subheader('GeoJSON Output')
        features = []

        for drawing in map_data['all_drawings']:
            feature = {
                "type": "Feature",
                "properties": {},
                "geometry": drawing
            }
            features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        # print("geojson_data ", type(geojson_data))
        # print("geojson_data ", geojson_data.keys())
        # print("geojson_data ", geojson_data['features'][0]['geometry'])
        coordinates = geojson_data['features'][0]['geometry']['geometry']['coordinates'][0]
        return coordinates

def get_satellite_image(vis_day, folder_image):
    dates = os.listdir(folder_image)
    sorted_dates = sorted(dates, key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    print(f"No. dates: {len(sorted_dates)}")
    print(sorted_dates)
    date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in sorted_dates]
    vis_day = datetime.strptime(vis_day, '%Y-%m-%d').date()
    closest_date = min(date_objects, key=lambda date: abs(date - vis_day))

def app():

    st.title('Demo for Satellite Project for Arkansas')
    map_data = get_map()
    start_date, end_date,vis_day = get_start_end_date()
    print("start_date ", start_date)
    print("end_date ", end_date)



    roig = get_roi(map_data)
    if st.button("Download"):
        with st.spinner('Wait for downloading satellite image from '+str(start_date)+' to '+str(end_date)):
            download_dataset(roig, start_date, end_date, config["save_satellite_dir"])
        st.success('Completed the dataset download!'+str(start_date)+' to '+str(end_date))

        # get_satellite_image(vis_day, config["save_satellite_dir"])

        with st.spinner('Wait for preprocessing satellite image from '+str(start_date)+' to '+str(end_date)):
            preprocess_AR(config["save_satellite_dir"], config["cdl_dir"], config["processed_dir"])
        st.success('Completed the dataset preprocessing!'+str(start_date)+' to '+str(end_date))

        with st.spinner('Wait for inference satellite image from '+str(start_date)+' to '+str(end_date)):
            inference_AR(config["model_config"],config["processed_dir"], config["output_vis"])
        st.success('Completed the dataset inference!'+str(start_date)+' to '+str(end_date))

        st.image(config["output_vis"], caption='Crop type segmentaion')








    #Download data 



    print("roig ", roig)


if __name__ == '__main__':
    app()
    print("Run the app for Arkansas")