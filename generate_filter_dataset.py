import cv2
import os
import glob
from tqdm import tqdm
import argparse
import numpy as np
import random

RUNNING_IN_COLAB = False

if RUNNING_IN_COLAB:
    from google.colab import drive
    DRIVE_MOUNT_POINT = "/content/drive"
    DEFAULT_BASE_PATH = os.path.join(DRIVE_MOUNT_POINT, "My Drive/adobe5k")
else:
    # Change this path to the base directory of your datasets accordingly
    DEFAULT_BASE_PATH = "path/to/your/local/datasets_folder"

DEFAULT_INPUT_SUBDIR = "raw"


def apply_oil_painting_filter(img, size=10, dyn_ratio=1):
    return cv2.xphoto.oilPainting(img, size, dyn_ratio)
def apply_pencil_sketch_filter(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
    _ , sketch_gray = cv2.pencilSketch(img, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
    return sketch_gray
def apply_grayscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def apply_sepia_filter(img):
    img_float = img.astype(np.float32)
    sepia_matrix = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    sepia_img_float = cv2.transform(img_float, sepia_matrix)
    return np.clip(sepia_img_float, 0, 255).astype(np.uint8)
def apply_cartoon_effect(img, d=9, sigma_color=250, sigma_space=250, block_size=9, c=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=block_size, C=c)
    color = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_color)
def apply_gaussian_blur(img, kernel_size_val=31):
    if kernel_size_val % 2 == 0: kernel_size_val += 1
    return cv2.GaussianBlur(img, (kernel_size_val, kernel_size_val), 0)

FILTER_FUNCTIONS = {
    "oil": apply_oil_painting_filter, "sketch": apply_pencil_sketch_filter,
    "grayscale": apply_grayscale_filter, "sepia": apply_sepia_filter,
    "cartoon": apply_cartoon_effect, "gaussian": apply_gaussian_blur,
}


def process_images(args):
    input_dir = os.path.join(args.base_path, args.input_subdir)
    if not args.output_subdir:
        args.output_subdir = f"{args.filter}_filtered"
    output_dir = os.path.join(args.base_path, args.output_subdir)
    
    print(f"--- Starting Filter Generation ---")
    print(f"Selected Filter: {args.filter}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    if not os.path.isdir(input_dir):
        print(f"\nError: Input directory not found at '{input_dir}'")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_extensions = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"No images found in {input_dir}.")
        return

    print(f"Found {len(image_paths)} total images in the source directory.")

    random.shuffle(image_paths)

    if args.num_images != -1 and args.num_images < len(image_paths):
        print(f"Limiting processing to a random subset of {args.num_images} images.")
        image_paths = image_paths[:args.num_images]

    print(f"Will process {len(image_paths)} images.")
    
    filter_function = FILTER_FUNCTIONS[args.filter]

    for img_path in tqdm(image_paths, desc=f"Applying {args.filter} filter"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"\nWarning: Could not read image {img_path}. Skipping.")
                continue
            
            params = {}
            if args.filter == 'oil':
                params = {'size': args.oil_size, 'dyn_ratio': args.oil_dyn_ratio}
            elif args.filter == 'sketch':
                params = {'sigma_s': args.sketch_sigma_s, 'sigma_r': args.sketch_sigma_r, 'shade_factor': args.sketch_shade_factor}
            elif args.filter == 'cartoon':
                params = {'d': args.cartoon_d, 'sigma_color': args.cartoon_sigma_color, 'sigma_space': args.cartoon_sigma_space, 'block_size': args.cartoon_block_size, 'c': args.cartoon_c}
            elif args.filter == 'gaussian':
                params = {'kernel_size_val': args.blur_kernel}

            filtered_img = filter_function(img, **params)

            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, filtered_img)

        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")

    print("\n--- Processing Complete ---")
    print(f"Filtered images saved in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply various filters to an image dataset.")
    
    parser.add_argument('--filter', type=str, required=True, choices=list(FILTER_FUNCTIONS.keys()), help="The type of filter to apply.")
    
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH, help="The root directory for your datasets.")
    parser.add_argument('--input_subdir', type=str, default=DEFAULT_INPUT_SUBDIR, help="Subdirectory containing raw images.")
    parser.add_argument('--output_subdir', type=str, default=None, help="Subdirectory to save filtered images. (Defaults to 'filter_name_filtered')")
    
    parser.add_argument('--num_images', type=int, default=-1, help="Maximum number of images to process. Default is -1, which means all images.")

    parser.add_argument('--oil_size', type=int, default=10, help="[Oil] Size of the neighborhood.")
    parser.add_argument('--oil_dyn_ratio', type=int, default=1, help="[Oil] Contrast parameter.")
    parser.add_argument('--sketch_sigma_s', type=int, default=60, help="[Sketch] Sigma_s parameter.")
    parser.add_argument('--sketch_sigma_r', type=float, default=0.07, help="[Sketch] Sigma_r parameter.")
    parser.add_argument('--sketch_shade_factor', type=float, default=0.05, help="[Sketch] Shade factor.")
    parser.add_argument('--cartoon_d', type=int, default=9, help="[Cartoon] Diameter of pixel neighborhood.")
    parser.add_argument('--cartoon_sigma_color', type=int, default=250, help="[Cartoon] Filter sigma in the color space.")
    parser.add_argument('--cartoon_sigma_space', type=int, default=250, help="[Cartoon] Filter sigma in the coordinate space.")
    parser.add_argument('--cartoon_block_size', type=int, default=9, help="[Cartoon] Block size for edge detection.")
    parser.add_argument('--cartoon_c', type=int, default=2, help="[Cartoon] Constant for edge detection.")
    parser.add_argument('--blur_kernel', type=int, default=31, help="[Gaussian] Kernel size for the blur. Must be an odd number.")

    args = parser.parse_args()
    
    if RUNNING_IN_COLAB:
        print("Running in Google Colab mode. Mounting Drive...")
        try:
            drive.mount(DRIVE_MOUNT_POINT, force_remount=True)
            print("Google Drive mounted successfully.")
        except Exception as e:
            print(f"Could not mount Google Drive. Aborting. Error: {e}")
            exit()
    else:
        print("Running in local mode.")
        if args.base_path == DEFAULT_BASE_PATH:
            print("\nWarning: Using the default placeholder path for local execution.")
            print(f"Please provide your actual dataset path using: --base_path 'C:/path/to/your/folder'")

    process_images(args)

