import os
import shutil 
import json
from PIL import Image
from collections import defaultdict

import ee
import geemap
import numpy as np
import rasterio
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from folium.plugins import Draw

from data.Arkansas.Download import download_dataset
from data.Arkansas.pre_processing import preprocess_AR
from train_and_eval.inference_AR24 import inference_AR


ee.Authenticate()
ee.Initialize(project='satelite-430703')

MAPBOX_TOKEN = "pk.eyJ1IjoidnVvbmdoIiwiYSI6ImNseWRobHVtZTA0a2EyaW9wc2loM2cxOWIifQ.U5utf4cO8ldi087Mn-h0FA"

config = {"save_satellite_dir": '/home/vuonghn/research/dataset/satellite/arkansas/satellite_images/2023/',
          "cdl_dir": "/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/",
          "processed_dir": "/home/vuonghn/research/dataset/satellite/arkansas/preprocessed_data/preprocessed_demo",
          "output_vis": './output/',
          "model_config": 'configs/Arkansas/TSViT_AR24.yaml',
          "ground_truth": '/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/rgb_27classes_cdl.tif'}


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
    map_data = st_folium(m, width=700, height=600)
    return map_data


def get_start_end_date():
    # Title for your app
    st.sidebar.title('Date Range')

    # Create a start date input widget
    start_date = st.sidebar.date_input("Select a start date:", datetime(2023, 1, 1))
    
    # Create an end date input widget
    # By default, set it to one week from the start date
    end_date = st.sidebar.date_input("Select an end date:", start_date + timedelta(days=364))
    vis_day = st.sidebar.date_input("Visualization date:", start_date + timedelta(days=30))

    # Check if the start date is after the end date and display a warning
    if start_date > end_date or vis_day > end_date or vis_day < start_date:
        st.sidebar.error('Error: Must Visualized day < .')

    # Display the selected date range5
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
    date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in sorted_dates]
    vis_day = datetime.strptime(vis_day, '%Y-%m-%d').date()
    closest_date = min(date_objects, key=lambda date: abs(date - vis_day))


def get_subregion_indices_in_roi(ar_roig, user_roi):
    """
    Given a predefined region (ar_roig) and a user-defined region of interest (user_roi),
    return the list of patch indices (i, j) that fall inside the user ROI on a 20x20 grid.
    
    Parameters:
        ar_roig (list): A list of four coordinate pairs defining the corners of the predefined region.
                        e.g., [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]
        user_roi (list): A list of four coordinate pairs defining the corners of the user-defined ROI.
    
    Returns:
        patch_indices (list): A list of tuples (i, j) representing the grid indices.
    """
    # Extract min and max for longitude and latitude of the predefined region
    lon_min_ar = min(ar_roig[0][0], ar_roig[1][0])
    lon_max_ar = max(ar_roig[2][0], ar_roig[3][0])
    lat_min_ar = min(ar_roig[1][1], ar_roig[3][1])
    lat_max_ar = max(ar_roig[0][1], ar_roig[2][1])

    # Create linspace for longitude and latitude for the predefined region
    lon_range_ar = np.linspace(lon_min_ar, lon_max_ar, 21)
    lat_range_ar = np.linspace(lat_min_ar, lat_max_ar, 21)

    # Compute the centers of the patches (grid points)
    lon_centers = (lon_range_ar[:-1] + lon_range_ar[1:]) / 2
    lat_centers = (lat_range_ar[:-1] + lat_range_ar[1:]) / 2

    # Create 2D meshgrid of center points
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Extract min and max for longitude and latitude of the user-defined ROI
    lon_min_user = min(user_roi[0][0], user_roi[1][0])
    lon_max_user = max(user_roi[2][0], user_roi[3][0])
    lat_min_user = min(user_roi[1][1], user_roi[3][1])
    lat_max_user = max(user_roi[0][1], user_roi[2][1])

    # Create a mask to find the indices of the patches inside the user-defined ROI
    mask = (lon_grid >= lon_min_user) & (lon_grid <= lon_max_user) & \
           (lat_grid >= lat_min_user) & (lat_grid <= lat_max_user)

    # Extract the indices where the mask is True
    patch_indices = np.argwhere(mask)

    # Convert to a list of tuples for consistency with previous approach
    patch_indices = [tuple(idx) for idx in patch_indices]

    return patch_indices


def find_closest_day(query_day, dates):
    query_day = datetime.strptime(query_day, '%Y-%m-%d')
    
    # Initialize variables to track the closest date and the smallest difference
    closest_date = None
    smallest_diff = float('inf')
    
    # Iterate over each date in the list
    for date in dates:
        # Convert the current date to a datetime object
        current_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Calculate the absolute difference in days
        diff = abs((current_date - query_day).days)
        
        # Update the closest date and smallest difference if a smaller difference is found
        if diff < smallest_diff:
            smallest_diff = diff
            closest_date = date
    
    return closest_date


def read_band_img(band_path, is_jpg):
    if is_jpg:
        return plt.imread(band_path)
    else:
        band_img = rasterio.open(band_path).read(1)
        band_img = (band_img - band_img.min()) / (band_img.max() - band_img.min())
        band_img = (plt.get_cmap('viridis')(band_img)[:, :, :3] * 255).astype(np.uint8)
        return band_img


def retrieve_band(subregion_indices, vis_day, band_type='TCI'):
    all_imgs = []
    if band_type == 'TCI':
        ext_type = 'jpg'
    else:
        ext_type = 'png'

    for lat_id, lon_id in subregion_indices:
        root_path = '/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/'
        subregion_path = os.path.join(root_path, f'{lon_id}_{lat_id}')
        dates = [day for day in os.listdir(subregion_path) if os.path.isdir(os.path.join(subregion_path, day))]
        closest_date = find_closest_day(vis_day, dates)
        band_path = os.path.join(subregion_path, closest_date, f'{band_type}_{closest_date}.{ext_type}')
        band_img = plt.imread(band_path)
        all_imgs.append([lat_id, lon_id, band_img])
        
    # Sort all_imgs first by lat_id in decreasing order, then by lon_id in increasing order
    all_imgs.sort(key=lambda x: (-x[0], x[1]))
        
    # Reorganize into a list of lists
    reorganized_imgs = []
    current_lat_id = None
    current_list = []
    
    for img in all_imgs:
        lat_id = img[0]
        if lat_id != current_lat_id:
            if current_list:
                reorganized_imgs.append(current_list)
            current_list = [img]
            current_lat_id = lat_id
        else:
            current_list.append(img)
    
    # Append the last group
    if current_list:
        reorganized_imgs.append(current_list)

    return reorganized_imgs

def app():
    # inference_AR(config["model_config"],config["processed_dir"], config["output_vis"], config["ground_truth"])
    st.title('Demo for Satellite Project for Arkansas')

    with st.container():

        map_data = get_map()
        with st.sidebar:
            start_date, end_date, vis_day = get_start_end_date()
            visualize_trigger = st.button("Visualize")
            predict_trigger = st.button("Analyze")

        ar_roig = [
            [-94.7610, 36.6652],
            [-94.7610, 32.8376],
            [-89.5522, 36.6652],
            [-89.5522, 32.8376],
        ]

        user_roig = get_roi(map_data)

        if visualize_trigger:
            subregion_indices = get_subregion_indices_in_roi(ar_roig, user_roig)
            all_imgs = retrieve_band(subregion_indices, vis_day)
            # Display the grid of images
            for img_row in all_imgs:
                cols = st.columns(len(img_row))
                for i, (lat_id, lon_id, img) in enumerate(img_row):
                    with cols[i]:
                        st.image(img, caption=f"Lat: {lat_id}, Lon: {lon_id}", use_column_width=True)

        if predict_trigger:
            subregion_indices = get_subregion_indices_in_roi(ar_roig, user_roig)
            with st.spinner('Performing inference satellite image from ' + str(start_date) + ' to ' + str(end_date)):
                #inference_AR(config["model_config"], config["processed_dir"], config["output_vis"], config["ground_truth"])
                config_file = 'configs/Arkansas/TSViT_AR23_infer.yaml'
                raw_dir = '/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/'
                input_dir = '/home/khoavo/Desktop/workplace/satelite/AR23_all/pickle24x24/'
                output_dir = './output_app/'
                os.makedirs(output_dir, exist_ok=True)
                for lat_id, lon_id in subregion_indices:
                    inference_AR(config_file, raw_dir, input_dir, f'{lon_id}_{lat_id}', output_dir, show_gt=False)
            st.success('Completed the dataset inference!'+str(start_date)+' to '+str(end_date))

            cols = st.columns(len(subregion_indices))
            for i, (lat_id, lon_id) in enumerate(subregion_indices):
                with cols[i]:
                    st.image(os.path.join(output_dir, f'{lon_id}_{lat_id}.png'), caption='Crop type prediction of ({lat_id}, {lon_id})')

        print("roig ", user_roig)


if __name__ == '__main__':
    app()