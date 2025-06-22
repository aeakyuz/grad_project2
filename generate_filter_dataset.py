import cv2
import os
import glob
from tqdm import tqdm
import argparse

# --- Import the centralized filter functions ---
from filter_library import FILTER_FUNCTIONS

# --- Main Configuration ---
# Set this to True if running on Google Colab, False if running locally
RUNNING_IN_COLAB = False

# --- Path Settings (Defaults) ---
if RUNNING_IN_COLAB:
    from google.colab import drive
    DRIVE_MOUNT_POINT = "/content/drive"
    DEFAULT_BASE_PATH = os.path.join(DRIVE_MOUNT_POINT, "My Drive/adobe5k")
else:
    DEFAULT_BASE_PATH = "path/to/your/local/datasets_folder"

DEFAULT_INPUT_SUBDIR = "raw"


def process_images(args):
    """
    Main function to process all images in a directory with the selected filter.
    """
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

    print(f"Found {len(image_paths)} images to process.")

    # Get the correct filter function from the library
    filter_function = FILTER_FUNCTIONS[args.filter]

    for img_path in tqdm(image_paths, desc=f"Applying {args.filter} filter"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"\nWarning: Could not read image {img_path}. Skipping.")
                continue
            
            # --- Simplified Logic: Build params and call the function ---
            params = {}
            if args.filter == 'oil':
                params = {'size': args.oil_size, 'dyn_ratio': args.oil_dyn_ratio}
            elif args.filter == 'sketch':
                params = {'sigma_s': args.sketch_sigma_s, 'sigma_r': args.sketch_sigma_r, 'shade_factor': args.sketch_shade_factor}
            elif args.filter == 'cartoon':
                params = {'d': args.cartoon_d, 'sigma_color': args.cartoon_sigma_color, 'sigma_space': args.cartoon_sigma_space, 'block_size': args.cartoon_block_size, 'c': args.cartoon_c}
            elif args.filter == 'gaussian':
                params = {'kernel_size_val': args.blur_kernel}

            # Call the selected function with its specific parameters
            filtered_img = filter_function(img, **params)

            # Save the result
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, filtered_img)

        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")

    print("\n--- Processing Complete ---")
    print(f"Filtered images saved in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply various filters to an image dataset.")
    
    parser.add_argument(
        '--filter', 
        type=str, 
        required=True, 
        choices=list(FILTER_FUNCTIONS.keys()), # Dynamically get choices
        help="The type of filter to apply."
    )
    
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH, help="The root directory for your datasets.")
    parser.add_argument('--input_subdir', type=str, default=DEFAULT_INPUT_SUBDIR, help="Subdirectory containing raw images.")
    parser.add_argument('--output_subdir', type=str, default=None, help="Subdirectory to save filtered images. (Defaults to 'filter_name_filtered')")
    
    # --- Filter-specific Arguments ---
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

