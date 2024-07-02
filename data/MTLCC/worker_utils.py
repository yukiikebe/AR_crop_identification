# Import necessary modules here, ensure they are also available in the environment where this will run
from data.MTLCC.tfrecord2tif import TFRecord2Numpy
import pickle
import numpy as np
import os

def extract_fun(paths, rootdir):
    # Initialize TFRecord2Numpy with absolute paths
    abs_in_paths = [os.path.join(rootdir, p) for p in paths]
    tfrec2np = TFRecord2Numpy(abs_in_paths)

    files_saved = []
    for i, p in enumerate(paths):
        try:
            relative_path = "%s.pickle" % "/".join(p.split("/")[-2:]).split(".")[0]
            save_name = os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl", relative_path)

            if os.path.exists(save_name):
                files_saved.append(os.path.join("data_IJGI18/datasets/full/240pkl", relative_path))
                print(f"Existing file {i + 1} of {len(paths)}")
                continue

            print(f"Processing file {i + 1} of {len(paths)}")
            x10, x20, x60, day, year, labels = tfrec2np.tfrecord2npy()
            year = np.ones(day.shape[0]).astype(np.int32) * (2000 + int(p.split("/")[-2][4:]))

            data = {"x10": x10.astype(np.int16),
                    "x20": x20.astype(np.int16),
                    "x60": x60.astype(np.int16),
                    "day": day.astype(np.int16),
                    "year": year.astype(np.int16),
                    "labels": labels.astype(np.int8)}

            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))

            with open(save_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            files_saved.append(os.path.join("data_IJGI18/datasets/full/240pkl", relative_path))

        except Exception as e:
            print(f"Error processing file {p}: {e}")
            continue

    return files_saved
