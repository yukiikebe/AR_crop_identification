import numpy as np


def get_patch_indices_in_roi(ar_roig, user_roi):
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


ar_roig = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]

#user_roig = get_roi(map_data)
user_roig = [
    [-94.262695, 35.639441],
    [-94.262695, 36.332828],
    [-93.339844, 36.332828],
    [-93.339844, 35.639441],
]

get_patch_indices_in_roi(ar_roig, user_roig)