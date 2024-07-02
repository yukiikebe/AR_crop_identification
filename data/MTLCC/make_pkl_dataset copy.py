from data.MTLCC.tfrecord2tif import TFRecord2Numpy
from utils.multiprocessing_utils import run_pool
import pickle
import numpy as np
import pandas as pd
import random
import os
from glob import glob
import argparse
from worker_utils import extract_fun
import multiprocessing







def tfrec2pickle(paths, rootdir):

    abs_in_paths = [os.path.join(rootdir, p) for p in paths]
    tfrec2np = TFRecord2Numpy(abs_in_paths)

    files_saved = []
    for i, p in enumerate(paths):
        # p = paths[0]

        try:
            relative_path = "%s.pickle" % "/".join(p.split("/")[-2:]).split(".")[0]
            save_name = os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl", relative_path)

            if os.path.exists(save_name):
                files_saved.append(os.path.join("data_IJGI18/datasets/full/240pkl", relative_path))
                print("existing file %d of %d" % (i, len(paths)))
                continue

            print("processing file %d of %d" % (i, len(paths)))
            x10, x20, x60, day, year, labels = tfrec2np.tfrecord2npy()
            year = np.ones(day.shape[0]).astype(np.int32) * (2000 + int(p.split("/")[-2][4:]))  # All tfrecords have year 2017

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

        except:
            continue

    return np.array(files_saved)


def split_data_paths(pkl_paths_df, train_ids_file, eval_ids_file, rootdir):
    # Define a lambda function to extract the ID from the file path
    get_id = lambda s: s.split("/")[-1].split(".")[0]

    # Load the evaluation and training IDs into DataFrames
    eval_ids = pd.read_csv(eval_ids_file, header=None)
    train_ids = pd.read_csv(train_ids_file, header=None)

    # Create a new column 'id' by applying 'get_id' to each path in the 'path' column
    pkl_paths_df['id'] = pkl_paths_df['path'].apply(get_id)

    # Filter the DataFrame based on whether the 'id' exists in the train or eval IDs
    train_paths = pkl_paths_df[pkl_paths_df['id'].isin(train_ids[0].astype(str))]
    eval_paths = pkl_paths_df[pkl_paths_df['id'].isin(eval_ids[0].astype(str))]

    # Save the filtered paths to CSV files
    train_paths[['path']].to_csv(os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl/train_paths.csv"),
                                 header=None, index=False)
    eval_paths[['path']].to_csv(os.path.join(rootdir, "data_IJGI18/datasets/full/240pkl/eval_paths.csv"),
                                header=None, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TFRecords into Pickle files.')
    parser.add_argument('--rootdir', required=True, help='Data root directory')
    parser.add_argument('--numworkers', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    # Prepare your data paths
    data_paths = glob(os.path.join(args.rootdir, "data_IJGI18/datasets/full/240/data16/*.tfrecord.gz"))
    data_paths = ["data_IJGI18/datasets/full/240/data16/" + os.path.basename(path) for path in data_paths]

    # Assuming extract_fun is adjusted to handle a subset of data_paths and placed in worker_utils.py
    with multiprocessing.Pool(processes=args.numworkers) as pool:
        results = pool.starmap(extract_fun, [(chunk.tolist(), args.rootdir) for chunk in np.array_split(data_paths, args.numworkers)])

    # Concatenate results and prepare for splitting
    pkl_paths = np.concatenate(results)
    pkl_paths_df = pd.DataFrame(pkl_paths, columns=['path'])

    # Define paths for train, eval, (and possibly test) ID files
    train_ids_file = os.path.join(args.rootdir, "data_IJGI18/datasets/full/240/tileids/train_fold0.tileids")
    eval_ids_file = os.path.join(args.rootdir, "data_IJGI18/datasets/full/240/tileids/eval.tileids")
    
    # Split and save paths
    split_data_paths(pkl_paths_df, train_ids_file, eval_ids_file, args.rootdir)

