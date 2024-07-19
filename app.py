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
ee.Authenticate()
ee.Initialize(project='ee-vvuonghn')

MAPBOX_TOKEN = "pk.eyJ1IjoidnVvbmdoIiwiYSI6ImNseWRobHVtZTA0a2EyaW9wc2loM2cxOWIifQ.U5utf4cO8ldi087Mn-h0FA"
save_path = './satellite_images/'
# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
# os.makedirs(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    return start_date, end_date, vis_day



def get_map():
    st.title('Demo for Satellite Project')
    st.title('Selection the ROI')
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



# @st.cache 
def download_satellite_image(roig,start_date,end_date):

    # # Define region of interest (ROI) as a list of coordinates
    # roig = [
    #     [-93.70856254511031, 35.43604890036842],
    #     [-93.70856254511031, 35.57542037748203],
    #     [-93.96375646096627, 35.57542037748203],
    #     [-93.96375646096627, 35.43604890036842],
    #     [-93.70856254511031, 35.43604890036842]
    # ]

    # Select Sentinel-2 Image Collection
    roi = ee.Geometry.Polygon(roig)
    # collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    #     .filterDate('2023-01-01', '2023-02-15') \
    #     .filterBounds(roi)\
    #     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(str(start_date), str(end_date)) \
        .filterBounds(roi)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    # Print the size of the S2 Image Collection
    collection_size = collection.size()
    print("The size of Sentinel-2 Image Collection:", collection_size.getInfo())

    # Print the attribute information of each image in the S2 Image Collection
    image_list = collection.toList(collection_size)



    print("Start downloading images...")
    for i in range(collection_size.getInfo()):
        image = ee.Image(image_list.get(i))
        date = image.date().format("YYYY-MM-dd").getInfo()
        cloud_percentage = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        print(f"Image {i+1}: Date={date}, Cloudy Pixel={cloud_percentage}%")

        # # Get bands with 10m resolution
        # bands_10m = ['B4', 'B3', 'B2']
        # image_10m = image.select(bands_10m)
        # description = f'rgb_{date}'
        # output_path = '/home/vuonghn/research/code/Agriculture/time-series-seg/data/' + description + '.tif'
        # geemap.ee_export_image(image_10m, filename=output_path, scale=10, crs='EPSG:3857', region=roi)
        # #geemap.ee_export_image_to_drive(image_10m, description=description, crs='EPSG:3857',folder='vvuonghn', region=roi, scale=10, maxPixels=1e13)

        # bands_10m = ['B4', 'B3', 'B2', 'B8']
        # image_10m = image.select(bands_10m)
        # description = f'10_image_{date}'
        # output_path = '/home/vuonghn/research/code/Agriculture/time-series-seg/data/' + description + '.tif'
        # geemap.ee_export_image(image_10m, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # geemap.ee_export_image_to_drive(image_10m, description=description, crs='EPSG:3857',folder='vvuonghn', region=roi, scale=10, maxPixels=1e13)
        # #

        bands_rgb = ['TCI_R','TCI_G','TCI_B']
        image_10m_ = image.select(bands_rgb)
        description = f'TCI_image10_{date}'
        output_path = save_path + description + '.tif'
        print("output_path ", output_path)
        geemap.ee_export_image(image_10m_, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # # geemap.ee_export_image_to_drive(image_10m_, description=description, crs='EPSG:3857',folder='vvuonghn', region=roi, scale=10, maxPixels=1e13)

        # # Get bands with 20m resolution
        # bands_20m = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        # image_20m = image.select(bands_20m)
        # description = f'20_image_{date}'
        # geemap.ee_export_image_to_drive(image_20m, description=description, crs='EPSG:3857',folder='vvuonghn', region=roi, scale=20, maxPixels=1e13)
        
        # SCL = ['SCL']
        # SCL_img = image.select(SCL)
        # description = f'SCL_image_{date}'
        # geemap.ee_export_image_to_drive(SCL_img, description=description, crs='EPSG:3857',folder='vvuonghn', region=roi, scale=20, maxPixels=1e13)

        # # Get bands with 60m resolution
        # bands_60m = ['B1', 'B9', 'QA60']
        # image_60m = image.select(bands_60m)
        # description = f'60_image_{date}'
        # geemap.ee_export_image_to_drive(image_60m, description=description,crs='EPSG:3857', folder='vvuonghn', region=roi, scale=60, maxPixels=1e13)
        
        
        # bands_20m = ['MSK_CLDPRB']
        # image_20m = image.select(bands_20m)
        # description = f'MSK_CLDPRB_image_{date}'
        # geemap.ee_export_image_to_drive(image_20m, description=description, crs='EPSG:3857',folder='image_new_ch', region=roi, scale=20, maxPixels=1e13)
    print("Done")
    return True

# Set your Mapbox access token here


# st.title('Demo for Satellite Project')
# st.title('Selection the ROI')
# # Initialize a Folium map using Mapbox tiles
# m = folium.Map(
#     location=[34.745092625799586, -92.28017348986518],
#     zoom_start=10,
#     tiles=f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
#     # tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
#     attr='Mapbox attribution'
# )

# # Add drawing options to the map
# from folium.plugins import Draw
# draw = Draw(export=True)
# draw.add_to(m)




# Function to handle GeoJSON output
def handle_geojson_output(map_data):
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
            # roig = [
    #     [-93.70856254511031, 35.43604890036842],
    #     [-93.70856254511031, 35.57542037748203],
    #     [-93.96375646096627, 35.57542037748203],
    #     [-93.96375646096627, 35.43604890036842],
    #     [-93.70856254511031, 35.43604890036842]
    # ]
        print("coordinates ", type(coordinates))
        print("coordinates ", coordinates)
        with st.spinner('Wait for downloading satellite image'):
            # time.sleep(5)
            download_satellite_image(coordinates)
        st.success('Completed the dataset download!')
        

        path_image = '/home/vuonghn/research/code/Agriculture/time-series-seg/output/'
        list_image = os.listdir(path_image)
        first_image = os.path.join(path_image,list_image[0])
        print("first_image ", first_image)
        tiff_image = Image.open(first_image)
        jpeg_image = tiff_image.convert("RGB")
        # png_image =first_image.replace('.tif', '.png')
        
        jpeg_image.save('/home/vuonghn/research/code/Agriculture/time-series-seg/output/output.jpg')
        st.image('/home/vuonghn/research/code/Agriculture/time-series-seg/output/output.png', caption='Satellite image')


        
        # image = Image.open(first_image)
        # st.image(image)

    #     st.json(geojson_data)
    #     st.download_button(
    #         label="Download GeoJSON",
    #         data=json.dumps(geojson_data, indent=2),
    #         file_name='data.geojson',
    #         mime='application/json'
    #     )
    # else:
    #     st.info("Draw polygons on the map to generate GeoJSON.")

# # Sidebar for location search
# st.sidebar.title("Location Search")
# query = st.sidebar.text_input("Enter a location (city, state, country):")

# # Initialize the geolocator
# geolocator = Nominatim(user_agent="geoapp")

# # Default location (Central USA)
# default_location = [34.745092625799586, -92.28017348986518]
# zoom_start = 10

# if query:
#     # Attempt to geocode the query
#     location = geolocator.geocode(query)
#     if location:
#         map_location = [location.latitude, location.longitude]
#         zoom_start = 12  # Zoom in closer for searched locations
#     else:
#         st.sidebar.error("Location not found. Showing default location.")
#         map_location = default_location
# else:
#     map_location = default_location

# # Update map view based on location search
# m.location = map_location
# m.zoom_start = zoom_start

# # Display the map with Streamlit
# map_data = st_folium(m)

# Handle GeoJSON output
# handle_geojson_output(map_data)


if __name__ == '__main__':
    downloaded = False
    map_data = get_map()
    start_date, end_date,vis_day = get_start_end_date()
    print("start_date ", start_date)
    print("end_date ", end_date)
    roig = get_roi(map_data)
    print("roig ", roig)
    if roig:
        with st.spinner('Wait for downloading satellite image from '+str(start_date)+' to '+str(end_date)):
            downloaded = download_satellite_image(roig,start_date,end_date)
        st.success('Completed the dataset download!'+str(start_date)+' to '+str(end_date))
        # roig == None
    print("roig", roig )
    print("downloaded ",downloaded)
    if downloaded:
        list_image = os.listdir(save_path)
        list_image = [x.split('.tif')[0].split('TCI_image10_')[1] for x in list_image if x.startswith('TCI_image10_')]

        vis_day_date = vis_day  # Assuming vis_day is already a datetime.date object

        # Convert datetime.datetime objects to datetime.date for comparison
        date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in list_image]

        # Find the date closest to vis_day
        closest_date = min(date_objects, key=lambda date: abs(date - vis_day_date))

        # Convert the closest date back to string if needed
        closest_date_str = closest_date.strftime('%Y-%m-%d')
        print("closest_date_str ", closest_date_str)
        image_name = 'TCI_image10_'+closest_date_str+'.tif'
        image_path = os.path.join(save_path,image_name)
        tiff_image = Image.open(image_path)
        jpeg_image = tiff_image.convert("RGB")
        jpeg_image.save(save_path+'/output.jpg')
        st.image(save_path+'/output.jpg', caption='Satellite image at the day '+str(vis_day))

        # print("closest_date_str ", closest_date_str)
        # print("list_image ", list_image)
        # first_image = os.path.join(save_path,list_image[0])
        # print("first_image ", first_image)
        # tiff_image = Image.open(first_image)
        # jpeg_image = tiff_image.convert("RGB")
        # # png_image =first_image.replace('.tif', '.png')
        
        # jpeg_image.save(save_path+'/output.jpg')
        # st.image(save_path+'/output.jpg', caption='Satellite image')


    # print("roig ", roig)
