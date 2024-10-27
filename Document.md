# Crop type segmentation

## Download data

## Preprocessing data
Please check file `data/Arkansas/preprocessing.py`
### Config

Please set the config path for `satellite_image_dir` and `output_dir`. The output data after running preprocessing will be stored at `output_dir`

Note: Please make sure that the file `configs/Arkansas/cdl.yaml` is available.

```
satellite_image_dir = "/data/datasets/satellite/raw_arkansas_2023/2023_all"
output_dir = "/data/datasets/satellite/AR23_processed"
```

The data structure for `satellite_image_dir` looks like that:

```
├── 0_0
    ├── 2023-01-03
        ├── 10m_rgb_2023-01-03.tif
        ├── B11_2023-01-03.tif
        ├── B12_2023-01-03.tif
        ├── B2_2023-01-03.tif
        ├── B3_2023-01-03.tif
        ├── B4_2023-01-03.tif
        ├── B5_2023-01-03.tif
        ├── B6_2023-01-03.tif
        ├── B7_2023-01-03.tif
        ├── B8_2023-01-03.tif
        ├── B8A_2023-01-03.tif
        ├── SCL_2023-01-03.tif
        ├── TCI_2023-01-03.jpg
        ├── TCI_B_2023-01-03.tif
        ├── TCI_G_2023-01-03.tif
        └── TCI_R_2023-01-03.tif
    ├── ...
    ├── 2023-12-27
    └── cdl.tif
├── 0_1
├── ...
└── 19_19

```

### Visualize

The function `visual_crop_distribution` will be used for the feature Visualize crop distribution. After calling this function, it will visualize crop distribution.

* `visual_crop_distribution(satellite_image_dir, output_dir)`
![Crop Distribution](doc/crop_distribution.png)
### Preprocessing data

The function `preprocess_satellite` processes large data. The input is the `satellite_image_dir` path as structured above. The output will be tiled data saved in a pickle file.


* For each pickle file, the format is:
    ```
    pickle_data = {
        'img': series_image, 
        'doy': doys,
        'labels': np.array(labels, dtype=np.uint8),
    }
    ```
* The API looks like `preprocess_satellite(satellite_image_dir, pickle_dir, num_cpus=8)`. Please change `num_cpus` based on your machine's resources.

### Split data
After run preprocessing data, the processed data will be stored, we need to split to `train/val`, we use random sampling based on the region. The figure bellow show the data for Arkansas region, the blue for validation and the green for training data. 

* Noted: In the future, we can change the way to do sampling data    
    ![grid_image](doc/grid_image.png)

Here is the log after running visual data and preprocessing data.
![preprocessing](doc/preprocessing.png)

The structure output data:

```
├── crop_distribution.png
├── fold-paths
    ├── train_sub_data.csv
    └── val_sub_data.csv
└── pickle24x24
    ├── 0_0
        ├── 984_744.pickle
        ├── ...
        ├── ...
        ├── ...
        └── 984_840.pickle
    ├── 0_1
    ├── ...
    └── 19_19

```
## Training models
* Setup the number of GPUs: 



## Evalution

## Inference and Visualzation
