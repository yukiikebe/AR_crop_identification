import argparse
import os
import random
import shutil
from unicodedata import category
from tqdm import tqdm

def transfer_file(src_path, dest_path, move_files=False):
    if move_files:
        shutil.move(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)


def collect_used_numeric_ids(root_dir, allowed_exts):
    used_ids = set()
    if not os.path.isdir(root_dir):
        return used_ids

    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.lower().endswith(allowed_exts):
                continue
            stem, _ = os.path.splitext(file_name)
            if stem.isdigit():
                used_ids.add(int(stem))

    return used_ids


def build_numeric_rename_map(file_names, output_root, allowed_exts, numeric_width=6):
    if numeric_width < 1:
        raise ValueError("numeric_width must be at least 1.")

    used_ids = collect_used_numeric_ids(output_root, allowed_exts)
    next_id = max(used_ids, default=0) + 1
    rename_map = {}

    for file_name in file_names:
        _, ext = os.path.splitext(file_name)
        while next_id in used_ids:
            next_id += 1
        rename_map[file_name] = f"{next_id:0{numeric_width}d}{ext.lower()}"
        used_ids.add(next_id)
        next_id += 1

    return rename_map

def split_dataset_by_matching_files(root_dir, output_dir, train_size, val_size, test_size):
    """
    Split datasets into train, validation, and test sets while ensuring the same files
    are selected across corresponding subdirectories (e.g., HR, LR, GT), handling prefixes like "patch_".

    Parameters:
        root_dir (str): Path to the dataset's root directory containing subdirectories like 'GT', 'HR', and 'LR'.
        output_dir (str): Path to save the split datasets.
        train_size (int): Number of images per category for the train set.
        val_size (int): Number of images per category for the validation set.
        test_size (int): Number of images per category for the test set.
    """
    total_required = train_size + val_size + test_size

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each category in the dataset
    # for category in os.listdir(os.path.join(root_dir, "GT")):  # Assume 'GT' contains the full category list
    for category in os.listdir(os.path.join(root_dir, "HR")):  # Assume 'HR' contains the full category list
        print(f"Processing category: {category}")

        # Collect matching files across HR, LR, and GT
        matching_files = set()  # Set of matching standardized names (without prefix)
        # file_mapping = {"HR": {}, "LR": {}, "GT": {}}  # Map standardized names to full paths
        file_mapping = {"HR": {}, "LR": {}}  # Map standardized names to full paths

        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR", "LR"]:
            category_path = os.path.join(root_dir, sub_dir, category)
            if not os.path.isdir(category_path):
                raise ValueError(f"Category {category} not found in subdirectory {sub_dir}. Check your directory structure.")

            # Process files in the category
            files = os.listdir(category_path)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Standardize file name (remove "patch_" prefix if present)
                    standardized_name = file_name[len("patch_"):] if file_name.startswith("patch_") else file_name
                    file_mapping[sub_dir][standardized_name] = os.path.join(category_path, file_name)

            # Add standardized names to matching set
            if not matching_files:
                matching_files = set(file_mapping[sub_dir].keys())
            else:
                matching_files = matching_files.intersection(file_mapping[sub_dir].keys())

        # Check if there are enough matching files
        matching_files = list(matching_files)
        if len(matching_files) < total_required:
            raise ValueError(f"Not enough matching files in category '{category}' (found {len(matching_files)}, needed {total_required}).")

        # Shuffle the matching files
        random.shuffle(matching_files)

        # Split files into train, validation, and test sets
        train_files = matching_files[:train_size]
        val_files = matching_files[train_size:train_size + val_size]
        test_files = matching_files[train_size + val_size:train_size + val_size + test_size]

        # Copy files into train, val, and test directories
        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR", "LR"]:
            for subset, subset_files in zip(
                ["train", "val", "test"],
                [train_files, val_files, test_files]
            ):
                subset_dir = os.path.join(output_dir, sub_dir, subset, category)
                os.makedirs(subset_dir, exist_ok=True)

                for standardized_name in tqdm(subset_files, desc=f"Copying {category} - {subset} ({sub_dir})"):
                    src_path = file_mapping[sub_dir][standardized_name]
                    dest_path = os.path.join(subset_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dest_path)

def split_dataset_by_matching_files_only_HR(root_dir, output_dir, train_size, val_size, test_size, move_files=False):
    """
    Split datasets into train, validation, and test sets while ensuring the same files
    are selected across corresponding subdirectories (e.g., HR, LR, GT), handling prefixes like "patch_".

    Parameters:
        root_dir (str): Path to the dataset's root directory containing subdirectories like 'GT', 'HR', and 'LR'.
        output_dir (str): Path to save the split datasets.
        train_size (int): Number of images per category for the train set.
        val_size (int): Number of images per category for the validation set.
        test_size (int): Number of images per category for the test set.
    """
    total_required = train_size + val_size + test_size

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each category in the dataset
    # for category in os.listdir(os.path.join(root_dir, "GT")):  # Assume 'GT' contains the full category list
    for category in os.listdir(os.path.join(root_dir, "HR")):  # Assume 'HR' contains the full category list
        print(f"Processing category: {category}")

        # Collect matching files across HR, LR, and GT
        matching_files = set()  # Set of matching standardized names (without prefix)
        # file_mapping = {"HR": {}, "LR": {}, "GT": {}}  # Map standardized names to full paths
        file_mapping = {"HR": {}}  # Map standardized names to full paths

        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR"]:
            category_path = os.path.join(root_dir, sub_dir, category)
            if not os.path.isdir(category_path):
                raise ValueError(f"Category {category} not found in subdirectory {sub_dir}. Check your directory structure.")

            # Process files in the category
            files = os.listdir(category_path)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    # Standardize file name (remove "patch_" prefix if present)
                    standardized_name = file_name[len("patch_"):] if file_name.startswith("patch_") else file_name
                    file_mapping[sub_dir][standardized_name] = os.path.join(category_path, file_name)

            # Add standardized names to matching set
            if not matching_files:
                matching_files = set(file_mapping[sub_dir].keys())
            else:
                matching_files = matching_files.intersection(file_mapping[sub_dir].keys())

        # Check if there are enough matching files
        matching_files = list(matching_files)
        if len(matching_files) < total_required:
            raise ValueError(f"Not enough matching files in category '{category}' (found {len(matching_files)}, needed {total_required}).")

        # Shuffle the matching files
        random.shuffle(matching_files)

        # Split files into train, validation, and test sets
        train_files = matching_files[:train_size]
        val_files = matching_files[train_size:train_size + val_size]
        test_files = matching_files[train_size + val_size:train_size + val_size + test_size]

        # Copy files into train, val, and test directories
        # for sub_dir in ["HR", "LR", "GT"]:
        for sub_dir in ["HR"]:
            for subset, subset_files in zip(
                ["train", "val", "test"],
                [train_files, val_files, test_files]
            ):
                subset_dir = os.path.join(output_dir, sub_dir, subset, category)
                os.makedirs(subset_dir, exist_ok=True)

                action = "Moving" if move_files else "Copying"
                for standardized_name in tqdm(subset_files, desc=f"{action} {category} - {subset} ({sub_dir})"):
                    src_path = file_mapping[sub_dir][standardized_name]
                    dest_path = os.path.join(subset_dir, os.path.basename(src_path))
                    transfer_file(src_path, dest_path, move_files=move_files)


def split_dataset_by_matching_files_ratio(root_dir, output_dir, train_ratio, val_ratio, test_ratio):
    """
    Split datasets into train, validation, and test sets by ratio, ensuring the same files
    are selected across corresponding subdirectories (e.g., HR, LR).

    Parameters:
        root_dir (str): Path to the dataset's root directory containing subdirectories like 'HR' and 'LR'.
        output_dir (str): Path to save the split datasets.
        train_ratio (float): Ratio of the train set (e.g., 0.8 for 80%).
        val_ratio (float): Ratio of the validation set (e.g., 0.1 for 10%).
        test_ratio (float): Ratio of the test set (e.g., 0.1 for 10%).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0.")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each category in the dataset
    for category in os.listdir(os.path.join(root_dir, "HR")):
        print(f"Processing category: {category}")
        
        # Collect matching files across HR and LR
        matching_files = set()  # Set of matching standardized names (without prefix)
        file_mapping = {"HR": {}, "LR": {}}
        
        for sub_dir in ["HR", "LR"]:
            category_path = os.path.join(root_dir, sub_dir, category)
            if not os.path.isdir(category_path):
                raise ValueError(f"Category {category} not found in subdirectory {sub_dir}. Check your directory structure.")
            
            # Process files in the category
            files = os.listdir(category_path)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    # Standardize file name (remove "patch_" prefix if present)
                    standardized_name = file_name
                    file_mapping[sub_dir][standardized_name] = os.path.join(category_path, file_name)
            
            # Add standardized names to matching set
            if not matching_files:
                matching_files = set(file_mapping[sub_dir].keys())
            else:
                matching_files = matching_files.intersection(file_mapping[sub_dir].keys())
            
        # Check if there are enough matching files
        matching_files = list(matching_files)
        total_files = len(matching_files)
        train_size = int(train_ratio * total_files)
        val_size = int(val_ratio * total_files)
        test_size = total_files - train_size - val_size
        
        if total_files < train_size + val_size + test_size:
            raise ValueError(f"Not enough matching files in category '{category}' (found {total_files}, needed {train_size + val_size + test_size}).")
        
        # Shuffle the matching files
        random.shuffle(matching_files)
        
        # Split files into train, validation, and test sets
        train_files = matching_files[:train_size]
        val_files = matching_files[train_size:train_size + val_size]
        test_files = matching_files[train_size + val_size:]
        
        # Copy files into train, val, and test directories
        for sub_dir in ["HR", "LR"]:
            for subset, subset_files in zip(
                ["train", "val", "test"],
                [train_files, val_files, test_files]
            ):
                subset_dir = os.path.join(output_dir, sub_dir, subset, category)
                os.makedirs(subset_dir, exist_ok=True)
                
                for standardized_name in tqdm(subset_files, desc=f"Copying {category} - {subset} ({sub_dir})"):
                    src_path = file_mapping[sub_dir][standardized_name]
                    dest_path = os.path.join(subset_dir, os.path.basename(src_path))
                    shutil.copy(src_path, dest_path)


def split_dataset_by_matching_files_ratio_only_HR(
    root_dir,
    output_dir,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=42,
    allowed_exts=('.jpg', '.jpeg', '.png', '.tif', '.tiff'),
    move_files=False,
    rename_numeric=False,
    numeric_width=6,
):
    """
    Split datasets into train/val/test by ratio using ONLY the HR branch.
    Category subfolders under HR are preserved in output.

    Directory assumptions:
        root_dir/HR/<category>/*.{jpg,jpeg,png,tif,tiff}

    Args:
        root_dir (str): Path to dataset root containing 'HR'.
        output_dir (str): Path to write 'HR/train|val|test/<category>/*'.
        train_ratio (float): e.g., 0.8
        val_ratio (float): e.g., 0.1
        test_ratio (float): e.g., 0.1
        seed (int): RNG seed for reproducibility.
        allowed_exts (tuple[str]): File extensions to accept.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio}).")

    if not os.path.isdir(root_dir):
        raise ValueError(f"root directory not found at: {root_dir}")

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(seed)

    all_files = [
        f for f in os.listdir(root_dir)
        if f.lower().endswith(allowed_exts) and os.path.isfile(os.path.join(root_dir, f))
    ]

    if len(all_files) == 0:
        raise ValueError(f"No files with extensions {allowed_exts} found in {root_dir}.")

    # Shuffle deterministically
    rng.shuffle(all_files)

    total_files = len(all_files)
    train_size = int(round(train_ratio * total_files))
    val_size   = int(round(val_ratio   * total_files))
    assigned = train_size + val_size
    test_size = total_files - assigned

    # Slices
    train_files = all_files[:train_size]
    val_files   = all_files[train_size:train_size + val_size]
    test_files  = all_files[train_size + val_size:]  # remainder

    rename_map = {}
    if rename_numeric:
        rename_map = build_numeric_rename_map(
            train_files + val_files + test_files,
            os.path.join(output_dir, "HR"),
            allowed_exts,
            numeric_width=numeric_width,
        )

    for subset_name, subset_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        if len(subset_files) == 0 and (subset_name == "test" and test_ratio == 0.0):
            # Skip creating empty test dir if test_ratio == 0
            continue

        subset_dir = os.path.join(output_dir, "HR", subset_name)
        os.makedirs(subset_dir, exist_ok=True)

        action = "Moving" if move_files else "Copying"
        for fname in tqdm(subset_files, desc=f"{action} {subset_name} (HR)"):
            src = os.path.join(root_dir, fname)
            output_name = rename_map.get(fname, fname)
            dst = os.path.join(subset_dir, output_name)
            transfer_file(src, dst, move_files=move_files)
                               

def build_parser():
    parser = argparse.ArgumentParser(description="Split an HR dataset into train/val/test folders.")
    parser.add_argument("--input-dir", default="../Train", help="Directory containing the source HR patch files.")
    parser.add_argument("--output-dir", default="../Train_tronto", help="Directory to write the split dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--transfer-mode",
        choices=("copy", "move"),
        default="move",
        help="Whether to copy files into split folders or move them.",
    )
    parser.add_argument(
        "--rename-numeric",
        action="store_true",
        help="Rename output files to unique numeric names such as 000001.tif.",
    )
    parser.add_argument(
        "--numeric-width",
        type=int,
        default=6,
        help="Zero-padding width used with --rename-numeric.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    split_dataset_by_matching_files_ratio_only_HR(
        args.input_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        seed=args.seed,
        move_files=(args.transfer_mode == "move"),
        rename_numeric=args.rename_numeric,
        numeric_width=args.numeric_width,
    )


if __name__ == "__main__":
    main()


# we can choose copy or move files
# python split_dataset.py \
#   --input-dir ../Train \
#   --output-dir ../Train_tronto \
#   --train-ratio 0.9 \
#   --val-ratio 0.1 \
#   --test-ratio 0.0 \
#   --transfer-mode move 

# --transfer-mode copy