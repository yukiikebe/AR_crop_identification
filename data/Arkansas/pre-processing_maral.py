import xarray as xr
import numpy as np
import glob
from skimage import exposure
import os
import rasterio
from typing import List,Dict
import xarray as xr
import cv2
import re
from datetime import datetime


# Define path mappings for different bands
PATH = {
    '10X': '10_', # 'B4', 'B3', 'B2', 'B8'
    '20X': '20_', # 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'
    '60X': '60_', # 'B1', 'B9', 'QA60'
    'SCL': 'SCL_', # 'SCL' # 20 
}
def create_splits(img, input_shape, pad_mode="reflect"):
    # A function to create splits out of a particular size from a given image.
    # images are split up row wise, i.e - row1 split up, row2 split up and so on
    # NOTE - padding is added in case the image can't be split into equal parts
    #       padding is added on the right and the bottom of the image, padding type
    #       is reflected by default
    splits = []

    # calculate pad length  
    pad_len_y = (0 - img.shape[0]) % input_shape[0]  
    pad_len_x = (0 - img.shape[1]) % input_shape[1]

    # option to pad the xarray
    if isinstance(img, xr.DataArray):
        img = img.pad(pad_width={
            "y": (0, pad_len_y), 
            "x": (0, pad_len_x)}, 
            mode=pad_mode
        )

    # option to pad normal numpy array
    else:
        img = np.pad(
            img, 
            [(0, pad_len_y), (0, pad_len_x), (0, 0)], 
            pad_mode
        )

    # loop through the indices with a step of input_shape
    # and create spltis for each index
    for i in range(0, img.shape[0], input_shape[0]):
        for j in range(0, img.shape[1], input_shape[1]):
          # use array slicing to select split size from the whole image
          splits.append(img[i : i + input_shape[0], j : j + input_shape[1]].copy())
    print("Done: create_splits  ", len(splits), img.shape)
    return splits, img

def get_rgb_from_s2(r, g, b):
    # normalize and stack r, g and b bands
    rgb = np.dstack([
        np.squeeze(r),  
        np.squeeze(g),
        np.squeeze(b),
    ])
    normalized_image = rgb.astype(np.float32) / 65535.0

    # clip between 0 and 1 to ensure we 
    # have no data out of range
#     rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


# from spliting import create_splits    



def read_cdl(item:str):
    '''
    read cdl
    '''
    path_cdl = "/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/"
    base_name = os.path.basename(item)
    cdl_name = base_name.replace('10_image_','rgb_').replace('.tif','_cdl.tif')
    cdl_path  = os.path.join(path_cdl,cdl_name)
    with rasterio.open(cdl_path) as src:
        image_data = src.read()
        num_bands = src.count
    return image_data

def read_data(path_base:str)->dict:
    """
    Reads and processes raster data for different bands.

    Args:
        path_base (str): The base path of the 10m resolution .tif file.

    Returns:
        dict: A dictionary where keys are band identifiers ('10X', '20X', '60X', 'SCL') and values 
              are tuples containing the image data array, and the x, y coordinates arrays.
    """
    print("path_base ", path_base)
    out = {}
    for band in PATH.values():
        # print("band ", band)

        base_name = os.path.basename(path_base)
        # print("base_name ", base_name)
        year,month ,day = base_name.split('_')[-1].split('-')
        day = day.replace('.tif','').replace('(1)','')
        day_of_year = convert_to_day_of_year(int(year),int(month) ,int(day))

        # print("year, month ,day : ", year, month ,day)
        # print("day_of_year ", day_of_year)
        
        base_name = path_base.replace('10_', band)
        # print("base_name ", base_name)
        with rasterio.open(base_name) as src:
            # Read the image data
            image_data = src.read()
            num_bands = src.count

            # print("num_bands ", num_bands)
            # print("image_data ", image_data.shape)
            # exit()
            width = src.width  # Width of the raster in pixels
            height = src.height
            transform = src.transform 
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x, y = (x * transform[0]) + transform[2], (y * transform[4]) + transform[5]
            out[band] = (image_data, x, y)
    return out,year,day_of_year

def convert_to_day_of_year(year, month, day):
    date = datetime(year, month, day)
    day_of_year = date.timetuple().tm_yday
    return day_of_year

def make_dict(band_dict:dict,cdl):
    '''
    Input:
        Dict X60_, X20_, X10_, SCL_
    Output:
        three Dataframes  
    '''
    img_10 = band_dict['10_'][0]
    _,w,h = img_10.shape
    scl    = band_dict['SCL_'][0]
    print("scl before shape: ",scl.shape)
    scl =np.squeeze(scl)
    print("scl after shape: ",scl.shape)
    # exit()
    upsampled_scl = cv2.resize(
        scl,
        (w,h),
        interpolation=cv2.INTER_NEAREST
    )
    print("upsampled_scl shape: ",upsampled_scl.shape)
    scl = np.moveaxis(upsampled_scl, [0, 1], [1, 0])
    print("upsampled_scl v1 shape: ",scl.shape)
    scl = np.expand_dims(scl, axis=0)
    print("upsampled_scl v2 shape: ",scl.shape)

    # scl before shape:  (1, 954, 1421)
    # scl after shape:  (954, 1421)
    # upsampled_scl shape:  (2842, 1908)
    # upsampled_scl v1 shape:  (1908, 2842)
    # upsampled_scl v2 shape:  (1, 1908, 2842)
    Red, Green, Blue, NIR = img_10[0],img_10[1],img_10[2],img_10[3]  # Band 2 (Red)
    S2_img_10  ={'red':Red,'green':Green,'blue':Blue,'NIR':NIR,'SCL':scl}
    # print("np.squeeze(S2_img_10['red']) ",np.squeeze(S2_img_10['red']).shape)
    # exit()
    data_stack_10 = np.dstack([
        np.squeeze(S2_img_10['red']),   #np.squeeze(S2_img_10['red'])  (1908, 2842)
        np.squeeze(S2_img_10['green']), 
        np.squeeze(S2_img_10['blue']), 
        np.squeeze(S2_img_10['NIR']), 
        np.squeeze(S2_img_10['SCL']), 
        np.squeeze(cdl),
    ])
    # (1908, 2842, 6)

     #'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'
    img_20 = band_dict['20_'][0]
    B5, B6, B7 ,B8A, B11,b12 = img_20[0],img_20[1],img_20[2],img_20[3],img_20[4],img_20[5]
    S2_img_20  ={'B5':B5,'B6':B6,'B7':B7,'B8A':B8A,'B11':B11,'B12':b12}
    data_stack_20 = np.dstack([
        np.squeeze(S2_img_20['B5']), 
        np.squeeze(S2_img_20['B6']), 
        np.squeeze(S2_img_20['B7']), 
        np.squeeze(S2_img_20['B8A']), 
        np.squeeze(S2_img_20['B11']),
        np.squeeze(S2_img_20['B12']),
        
    ])
    
    #['B1', 'B9', 'B10']
    img_60  = band_dict['60_'][0]
    B1,B9,B10 = img_60[0], img_60[1],img_60[2]   
    S2_img_60  ={'B1':B1,'B9':B9,'B10':B10}
    data_stack_60 = np.dstack([
        np.squeeze(S2_img_60['B1']), 
        np.squeeze(S2_img_60['B9']), 
        np.squeeze(S2_img_60['B10']),  
    ])
    data_stack_set =[]
    for i, stack in enumerate([data_stack_10, data_stack_20, data_stack_60]):
        print(i)
        if i==0:
            x,y = band_dict['10_'][1],band_dict[
                '10_'][2]
            # print("x y ",x.shape,y.shape)
            # exit()
        elif i==1:
            x,y = band_dict['20_'][1],band_dict['20_'][2]
        else:
            x,y = band_dict['60_'][1],band_dict['60_'][2]
        datastack = xr.DataArray(
        stack,
        dims=['y', 'x', 'bands'], 
        coords={
            'y': y[:, 0],  # y coordinates of each pixel
            'x': x[0, :],
            'bands': list(range(stack.shape[2]))

        }
        )
        data_stack_set.append(datastack)
    return data_stack_set
   
        
def splite_data(data_stack_set):
    '''
    tiling
    input:
    output

    '''
    data_stack_xr_10, data_stack_20, data_stack_60  = data_stack_set
    N = 24 #@param {type:"number"}
    splits_10, padded_img_10 = create_splits(data_stack_xr_10, [N, N])
    N = 12 #@param {type:"number"}
    splits_20, padded_img_20 = create_splits(data_stack_20, [N, N])
    N = 4
    splits_60, padded_img_60 = create_splits(data_stack_60, [N, N])


    num_splits_20 = len(splits_20)
    num_splits_60 = len(splits_60)
    print(num_splits_20,num_splits_60)
    return splits_10, splits_20, splits_60




def extract_date(filepath):
    # Extract the date using regex
    match = re.search(r'(\d{4}-\d{2}-\d{2})(?:\(\d+\))?\.tif$', filepath)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y-%m-%d')
    return datetime.min  # Return minimum date if no match found


def cloud_cover_check(ds_array):
    # Check for cloudy data in SCL band
    ds_array = np.squeeze(ds_array)
    masked = np.zeros((ds_array.shape[0], ds_array.shape[1]))
    masked[ds_array == 3] = 1
    masked[(ds_array >= 8)] = 1
    percent_valid = (1 - sum(sum(masked)) / (ds_array.shape[0] * ds_array.shape[1])) * 100

    return percent_valid
def nan_perc_check(ds_array):
    # Check for NaN data using the SCL band for S2
    ds_array = np.squeeze(ds_array)
    num_nan = np.isnan(ds_array).astype(int).sum()

    return (num_nan / (ds_array.shape[0] * ds_array.shape[1])) * 100
def filter_date(a,series,dep_lr,tr_shoul):
    id_lst =[]
    cloud =100- cloud_cover_check(a[series][dep_lr,:,:,-2])
    #Red
    value_red = nan_perc_check(a[series][dep_lr,:,:,0])
    if cloud<=10 or value_red<=0:
        pass
    return 
def check_data(data,serie_number,dep_lr_number,cloud_tr):
    id_lst =[]
    cloud = 100 - cloud_cover_check(data[serie_number][dep_lr_number,:,:,-2])
    value_red = nan_perc_check(data[serie_number][dep_lr_number,:,:,0])
    print(f'cloud percentage:{cloud}','pixel value:{value_red}')
    if cloud<=cloud_tr and value_red<=0:
        return True

def main():


    # Get list of all .tif files in the specified directory
    all_tif_10 = glob.glob('/home/vuonghn/research/dataset/satellite/arkansas/org_maral/images/10_*.tif')
    list_tile_10 =[]
    list_tile_20 =[]
    list_tile_60 =[]
    year_list =[]
    day_of_year_list =[]

    sorted_files = sorted(all_tif_10, key=extract_date)

    print(f"Found {len(sorted_files)} .tif files")

    # print("sorted_files ", sorted_files)
    # exit()


    for item in sorted_files:
        # if os. item
        try:
            print(os.path.basename(item),"*****")
            g,year,day_of_year= read_data(item)
            year_list.append(year)
            day_of_year_list.append(day_of_year)
            cdl = read_cdl(item)
            print("HERE")
            data_stack_set = make_dict(g,cdl)
            # if "10_image_2023-08-24.tif" in item:
            #     print("zakhire")
            #     data_stack_set1 =data_stack_set


            splits_10, splits_20, splits_60 = splite_data(data_stack_set)
            exit()
            list_tile_10.append(splits_10)
            list_tile_20.append(splits_20)
            list_tile_60.append(splits_60)
        except Exception as e:
            print(e)
            continue
            
        # print(os.path.basename(item),"*****")
        # g,year,day_of_year= read_data(item)
        # year_list.append(year)
        # day_of_year_list.append(day_of_year)
        # cdl = read_cdl(item)
        # data_stack_set = make_dict(g,cdl)
        # if "10_image_2023-08-24.tif" in item:
        #     print("zakhire")
        #     data_stack_set1 =data_stack_set
            
            
        # splits_10, splits_20, splits_60 = splite_data(data_stack_set)
        # list_tile_10.append(splits_10)
        # list_tile_20.append(splits_20)
        # list_tile_60.append(splits_60)
    numpy_array_10 = np.array(list_tile_10)
    numpy_array_20 = np.array(list_tile_20)
    numpy_array_60 = np.array(list_tile_60)

    print("here")
    print(numpy_array_10.shape)
    print(numpy_array_20.shape)
    print(numpy_array_60.shape)
    a = numpy_array_10.reshape(9520, 5, 24, 24, 6)
    series_number, depth, h,w,ch = a.shape

    numpy_array_10 = numpy_array_10.reshape(9520, 5, 24, 24, 6)
    numpy_array_20 = numpy_array_20.reshape(9520, 5, 12, 12, 6)
    numpy_array_60 = numpy_array_60.reshape(9520, 5, 4, 4, 3)

    # series_number = numpy_array_10.shape[1]
    import pickle
    tr_cloud = 10
    tr_miss  = 0
    cloud_tr = 0
    day = []
    year = []
    res_10 = []
    res_20 = []
    res_60 = []

    for series in range(series_number):
        for dep_lr in range(depth):
            if check_data(a,series,dep_lr,cloud_tr):
                day.append(day_of_year_list[dep_lr])
                year.append(year_list[dep_lr])
                res_10.append(numpy_array_10[series][dep_lr,:,:,:])
                res_20.append(numpy_array_20[series][dep_lr,:,:,:])
                res_60.append(numpy_array_60[series][dep_lr,:,:,:])

        res_10 =np.array(res_10)
        res_20 =np.array(res_20)
        res_60 =np.array(res_60)
        data = {
        'x10': res_10.astype('int16'),
        'x20': res_20.astype('int16'),
        'x60': res_60.astype('int16'),
        'day': day,
        'year':year,
        'labels': np.squeeze(numpy_array_10[:,1,:,:,4:].astype('int8')),
            }
        with open(f'/home/vuonghn/research/dataset/satellite/arkansas/dataset_vuonghn/{series}.pickle', 'wb') as file:
            pickle.dump(data, file)
        res_10 = []
        res_20 = []
        res_60 = []

if __name__ == "__main__":

    main()