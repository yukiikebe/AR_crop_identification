import argparse
import os
import re

import cv2
import numpy as np
from PIL import Image
import rasterio


def natural_key(value):
    return [int(token) if token.isdigit() else token for token in re.findall(r"\d+|\D+", value)]


def resolve_patch_step(patch_size, overlap):
    step = int(patch_size * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap must be less than 1.0")
    return step


def build_patch_positions(length, patch_size, step):
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, step))
    final_position = length - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def build_patch_coordinates(
    whole_image_size,
    patch_size,
    step,
    patch_rows_per_block=8,
    patch_cols_per_block=8,
):
    y_positions = build_patch_positions(whole_image_size[0], patch_size, step)
    x_positions = build_patch_positions(whole_image_size[1], patch_size, step)

    coordinates = []
    for row_start in range(0, len(y_positions), patch_rows_per_block):
        y_block = y_positions[row_start : row_start + patch_rows_per_block]
        for col_start in range(0, len(x_positions), patch_cols_per_block):
            x_block = x_positions[col_start : col_start + patch_cols_per_block]
            for y in y_block:
                for x in x_block:
                    coordinates.append((y, x))
    return coordinates


def autocrop_black(image_path, output_path, threshold=5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)

    if coords.size == 0:
        print("No cropping performed!")
        cv2.imwrite(output_path, img)
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img[y0:y1, x0:x1]
    cv2.imwrite(output_path, cropped)
    print(f"Cropped saved to {output_path}")


def build_whole_image_size(reference_image, patch_size, scale_multiplier=1, channels=None):
    with rasterio.open(reference_image) as src:
        height = int(src.height * scale_multiplier)
        width = int(src.width * scale_multiplier)
        count = src.count if channels is None else channels

    return (height, width, count)


def build_merge_canvas_size(reference_image, patch_size, overlap=0.0, scale_multiplier=1, channels=None):
    height, width, count = build_whole_image_size(
        reference_image=reference_image,
        patch_size=patch_size,
        scale_multiplier=scale_multiplier,
        channels=channels,
    )
    step = resolve_patch_step(patch_size, overlap)
    num_rows = max(1, (height + step - 1) // step)
    num_cols = max(1, (width + step - 1) // step)
    padded_height = (num_rows - 1) * step + patch_size
    padded_width = (num_cols - 1) * step + patch_size
    return (padded_height, padded_width, count)


def detect_scale_multiplier_hint(
    selected_patch_count,
    reference_image,
    patch_size,
    overlap,
    channels,
    current_scale_multiplier,
    candidate_scales=(2, 4, 8),
):
    if reference_image is None:
        return None

    step = resolve_patch_step(patch_size, overlap)
    matching_scales = []
    for scale in candidate_scales:
        if scale == current_scale_multiplier:
            continue

        scaled_size = build_merge_canvas_size(
            reference_image=reference_image,
            patch_size=patch_size,
            overlap=overlap,
            scale_multiplier=scale,
            channels=channels,
        )
        scaled_expected_count = len(build_patch_coordinates(scaled_size, patch_size, step))
        if scaled_expected_count == selected_patch_count:
            matching_scales.append(scale)

    if not matching_scales:
        return None

    if len(matching_scales) == 1:
        return (
            f" Hint: selected patch count matches --scale-multiplier {matching_scales[0]} "
            f"for this reference image."
        )

    scales = ", ".join(str(scale) for scale in matching_scales)
    return f" Hint: selected patch count matches these scale multipliers for this reference image: {scales}."


def derive_reference_stem(reference_image):
    return os.path.splitext(os.path.basename(reference_image))[0]


def filter_patch_files_by_name(patch_files, file_substring=None, reference_stem=None):
    if file_substring is not None:
        filtered_files = [f for f in patch_files if file_substring in f]
        if not filtered_files:
            raise ValueError(
                f"No patch files matched --file-substring {file_substring!r}. "
                f"Available files in the directory do not contain that token."
            )
        return filtered_files, file_substring

    if reference_stem:
        filtered_files = [f for f in patch_files if reference_stem in f]
        if filtered_files:
            return filtered_files, reference_stem

    return patch_files, None


def collect_patch_files(
    input_dir,
    extension,
    skip_first=0,
    max_patches=None,
    file_substring=None,
    reference_stem=None,
):
    patch_files = sorted([f for f in os.listdir(input_dir) if f.endswith(extension)], key=natural_key)
    if patch_files:
        filtered_files, applied_filter = filter_patch_files_by_name(
            patch_files,
            file_substring=file_substring,
            reference_stem=reference_stem,
        )
        selected_files = filtered_files[skip_first:]
        if max_patches is not None:
            selected_files = selected_files[:max_patches]
        return patch_files, filtered_files, selected_files, applied_filter

    alt_extension = ".png" if extension == ".tif" else ".tif"
    alt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(alt_extension)], key=natural_key)
    if alt_files:
        raise ValueError(
            f"No {extension.upper()} patches found in {input_dir}. "
            f"Found {len(alt_files)} {alt_extension.upper()} files instead; "
            f"did you mean to use the merge-{'png' if alt_extension == '.png' else 'tiff'} command?"
        )

    raise ValueError(f"No {extension.upper()} patches found in {input_dir}")


def validate_patch_count(
    patch_files,
    expected_count,
    input_dir,
    patch_size,
    overlap,
    skip_first,
    max_patches,
    applied_filter=None,
    extra_hint=None,
):
    num_patches = len(patch_files)
    if num_patches != expected_count:
        limit_text = "all patches" if max_patches is None else f"at most {max_patches} patches"
        filter_text = "" if applied_filter is None else f" Active filename filter: {applied_filter!r}."
        hint_text = "" if extra_hint is None else extra_hint
        raise ValueError(
            f"Patch count mismatch for {input_dir}: selected {num_patches} patches but expected {expected_count} "
            f"for patch_size={patch_size} and overlap={overlap}. "
            f"This usually means the folder mixes multiple patch sets, contains stale files, or uses a different "
            f"reference image/overlap. Current slice: skip_first={skip_first}, max_patches={limit_text}.{filter_text}"
            f"{hint_text}"
        )


def merge_tiff_patches(
    input_dir,
    output_png_path,
    whole_image_size,
    patch_size=256,
    overlap=0.0,
    skip_first=0,
    max_patches=None,
    file_substring=None,
    reference_stem=None,
    reference_image=None,
    channels=None,
    scale_multiplier=1.0,
):
    _, _, patch_files, applied_filter = collect_patch_files(
        input_dir,
        ".tif",
        skip_first=skip_first,
        max_patches=max_patches,
        file_substring=file_substring,
        reference_stem=reference_stem,
    )
    step = resolve_patch_step(patch_size, overlap)
    patch_coordinates = build_patch_coordinates(whole_image_size, patch_size, step)
    expected_count = len(patch_coordinates)
    scale_hint = None
    if len(patch_files) != expected_count:
        scale_hint = detect_scale_multiplier_hint(
            selected_patch_count=len(patch_files),
            reference_image=reference_image,
            patch_size=patch_size,
            overlap=overlap,
            channels=channels,
            current_scale_multiplier=scale_multiplier,
        )
    validate_patch_count(
        patch_files,
        expected_count,
        input_dir,
        patch_size,
        overlap,
        skip_first,
        max_patches,
        applied_filter=applied_filter,
        extra_hint=scale_hint,
    )
    num_patches = len(patch_files)

    whole_image_sum = np.zeros(whole_image_size, dtype=np.float64)
    whole_image_count = np.zeros(whole_image_size, dtype=np.float32)
    patch_idx = 0

    for y, x in patch_coordinates:
        if patch_idx >= num_patches:
            break

        patch_path = os.path.join(input_dir, patch_files[patch_idx])
        with rasterio.open(patch_path) as patch:
            patch_data_tiff = np.moveaxis(patch.read(), 0, -1)
            patch_h, patch_w, patch_c = patch_data_tiff.shape
            y_end = min(y + patch_h, whole_image_size[0])
            x_end = min(x + patch_w, whole_image_size[1])
            patch_slice = patch_data_tiff[: y_end - y, : x_end - x]

            if patch_c == whole_image_size[2]:
                whole_image_sum[y:y_end, x:x_end, :] += patch_slice
                whole_image_count[y:y_end, x:x_end, :] += 1.0
            else:
                whole_image_sum[y:y_end, x:x_end, :patch_c] += patch_slice
                whole_image_count[y:y_end, x:x_end, :patch_c] += 1.0

        patch_idx += 1

    safe_count = np.where(whole_image_count == 0, 1.0, whole_image_count)
    whole_image = whole_image_sum / safe_count

    with rasterio.open(os.path.join(input_dir, patch_files[0])) as first_patch:
        output_dtype = np.dtype(first_patch.dtypes[0])

    if np.issubdtype(output_dtype, np.integer):
        dtype_info = np.iinfo(output_dtype)
        whole_image = np.clip(np.rint(whole_image), dtype_info.min, dtype_info.max).astype(output_dtype)
    else:
        whole_image = whole_image.astype(output_dtype)

    whole_image_png = None
    if whole_image.shape[2] >= 3:
        whole_image_png = whole_image[:, :, :3]
        if whole_image_png.dtype != np.uint8:
            img_min, img_max = whole_image_png.min(), whole_image_png.max()
            if img_max > img_min:
                whole_image_png = ((whole_image_png - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                whole_image_png = np.zeros_like(whole_image_png, dtype=np.uint8)

    output_tif_path = output_png_path.replace(".png", ".tif")
    print("Final image shape:", whole_image.shape)
    with rasterio.open(
        output_tif_path,
        "w",
        driver="GTiff",
        height=whole_image.shape[0],
        width=whole_image.shape[1],
        count=whole_image.shape[2],
        dtype=whole_image.dtype,
    ) as dst:
        for band in range(whole_image.shape[2]):
            dst.write(whole_image[:, :, band], band + 1)

    if whole_image_png is not None:
        Image.fromarray(whole_image_png, "RGB").save(output_png_path)
        print("save png image")


def merge_png_patches(
    input_dir,
    output_png_path,
    whole_image_size,
    patch_size=256,
    overlap=0.0,
    skip_first=0,
    max_patches=None,
    file_substring=None,
    reference_stem=None,
    reference_image=None,
    channels=3,
    scale_multiplier=1.0,
):
    _, _, patch_files, applied_filter = collect_patch_files(
        input_dir,
        ".png",
        skip_first=skip_first,
        max_patches=max_patches,
        file_substring=file_substring,
        reference_stem=reference_stem,
    )
    step = resolve_patch_step(patch_size, overlap)
    patch_coordinates = build_patch_coordinates(whole_image_size, patch_size, step)
    expected_count = len(patch_coordinates)
    scale_hint = None
    if len(patch_files) != expected_count:
        scale_hint = detect_scale_multiplier_hint(
            selected_patch_count=len(patch_files),
            reference_image=reference_image,
            patch_size=patch_size,
            overlap=overlap,
            channels=channels,
            current_scale_multiplier=scale_multiplier,
        )
    validate_patch_count(
        patch_files,
        expected_count,
        input_dir,
        patch_size,
        overlap,
        skip_first,
        max_patches,
        applied_filter=applied_filter,
        extra_hint=scale_hint,
    )
    print("count patch files", len(patch_files))
    if applied_filter is not None:
        print(f"Applied filename filter: {applied_filter}")
    num_patches = len(patch_files)

    whole_image_sum = np.zeros((whole_image_size[0], whole_image_size[1], 3), dtype=np.float32)
    whole_image_count = np.zeros((whole_image_size[0], whole_image_size[1]), dtype=np.uint16)
    patch_idx = 0

    for y, x in patch_coordinates:
        if patch_idx >= num_patches:
            break

        patch_path = os.path.join(input_dir, patch_files[patch_idx])
        patch = Image.open(patch_path).convert("RGB")
        patch_data = np.array(patch)

        if patch_data.shape[:2] != (patch_size, patch_size):
            print(f"Warning: Patch {patch_path} has unexpected size {patch_data.shape[:2]}")
            patch_data = np.array(patch.resize((patch_size, patch_size)))

        patch_h, patch_w = patch_data.shape[:2]
        y_end = min(y + patch_h, whole_image_size[0])
        x_end = min(x + patch_w, whole_image_size[1])
        patch_slice = patch_data[: y_end - y, : x_end - x]

        whole_image_sum[y:y_end, x:x_end, :] += patch_slice.astype(np.float32, copy=False)
        whole_image_count[y:y_end, x:x_end] += 1
        patch_idx += 1

    safe_count = whole_image_count.astype(np.float32)
    safe_count[safe_count == 0] = 1.0
    whole_image_sum /= safe_count[:, :, None]
    np.clip(np.rint(whole_image_sum), 0, 255, out=whole_image_sum)
    whole_image = whole_image_sum.astype(np.uint8)

    Image.fromarray(whole_image, "RGB").save(output_png_path)
    print(f"Merged image saved to {output_png_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Merge image patches or crop black borders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_png_parser = subparsers.add_parser("merge-png", help="Merge PNG patches back into a whole PNG image.")
    merge_png_parser.add_argument("--input-dir", required=True, help="Directory containing PNG patches.")
    merge_png_parser.add_argument("--output-path", required=True, help="Output PNG path.")
    merge_png_parser.add_argument("--reference-image", required=True, help="Reference TIFF used to infer image size.")
    merge_png_parser.add_argument("--patch-size", type=int, default=256, help="Patch size used during extraction.")
    merge_png_parser.add_argument("--overlap", type=float, default=0.0, help="Patch overlap ratio used during extraction.")
    merge_png_parser.add_argument("--scale-multiplier", type=float, default=1.0, help="Scale reference size before merge.")
    merge_png_parser.add_argument("--channels", type=int, default=3, help="Output channel count for the merged canvas.")
    merge_png_parser.add_argument("--skip-first", type=int, default=0, help="Skip this many sorted patch files before merging.")
    merge_png_parser.add_argument("--max-patches", type=int, default=None, help="Merge at most this many sorted patch files.")
    merge_png_parser.add_argument("--file-substring", default=None, help="Only merge patch files whose names contain this substring.")

    merge_tiff_parser = subparsers.add_parser("merge-tiff", help="Merge TIFF patches back into a whole TIFF image.")
    merge_tiff_parser.add_argument("--input-dir", required=True, help="Directory containing TIFF patches.")
    merge_tiff_parser.add_argument("--output-path", required=True, help="Output PNG path; TIFF is saved beside it.")
    merge_tiff_parser.add_argument("--reference-image", required=True, help="Reference TIFF used to infer image size.")
    merge_tiff_parser.add_argument("--patch-size", type=int, default=256, help="Patch size used during extraction.")
    merge_tiff_parser.add_argument("--overlap", type=float, default=0.0, help="Patch overlap ratio used during extraction.")
    merge_tiff_parser.add_argument("--scale-multiplier", type=float, default=1.0, help="Scale reference size before merge.")
    merge_tiff_parser.add_argument("--channels", type=int, default=None, help="Override channel count.")
    merge_tiff_parser.add_argument("--skip-first", type=int, default=0, help="Skip this many sorted patch files before merging.")
    merge_tiff_parser.add_argument("--max-patches", type=int, default=None, help="Merge at most this many sorted patch files.")
    merge_tiff_parser.add_argument("--file-substring", default=None, help="Only merge patch files whose names contain this substring.")

    crop_parser = subparsers.add_parser("autocrop-black", help="Crop black borders from a PNG/JPG image.")
    crop_parser.add_argument("--image-path", required=True, help="Input image path.")
    crop_parser.add_argument("--output-path", required=True, help="Output image path.")
    crop_parser.add_argument("--threshold", type=int, default=5, help="Dark pixel threshold used for cropping.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "merge-png":
        reference_stem = derive_reference_stem(args.reference_image)
        whole_image_size = build_merge_canvas_size(
            reference_image=args.reference_image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            scale_multiplier=args.scale_multiplier,
            channels=args.channels,
        )
        merge_png_patches(
            args.input_dir,
            args.output_path,
            whole_image_size,
            patch_size=args.patch_size,
            overlap=args.overlap,
            skip_first=args.skip_first,
            max_patches=args.max_patches,
            file_substring=args.file_substring,
            reference_stem=reference_stem,
            reference_image=args.reference_image,
            channels=args.channels,
            scale_multiplier=args.scale_multiplier,
        )
    elif args.command == "merge-tiff":
        reference_stem = derive_reference_stem(args.reference_image)
        whole_image_size = build_merge_canvas_size(
            reference_image=args.reference_image,
            patch_size=args.patch_size,
            overlap=args.overlap,
            scale_multiplier=args.scale_multiplier,
            channels=args.channels,
        )
        merge_tiff_patches(
            args.input_dir,
            args.output_path,
            whole_image_size,
            patch_size=args.patch_size,
            overlap=args.overlap,
            skip_first=args.skip_first,
            max_patches=args.max_patches,
            file_substring=args.file_substring,
            reference_stem=reference_stem,
            reference_image=args.reference_image,
            channels=args.channels,
            scale_multiplier=args.scale_multiplier,
        )
    elif args.command == "autocrop-black":
        autocrop_black(args.image_path, args.output_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
