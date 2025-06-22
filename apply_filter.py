import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os
import time
import cv2  # Import OpenCV for padding
from tqdm import tqdm # For a progress bar

def _pyramid_window(patch_size):
    """
    Creates a 2D pyramid-shaped blending window.
    """
    # Create a 1D linear ramp from a small value to 1, then back to a small value
    half = patch_size // 2
    ramp_up = np.linspace(1 / half, 1.0, half, endpoint=False)
    ramp_down = np.linspace(1.0, 1 / half, patch_size - half, endpoint=False)
    window_1d = np.concatenate([ramp_up, ramp_down])
    # Create the 2D window by taking the outer product
    window_2d = np.outer(window_1d, window_1d)
    # Reshape to (patch_size, patch_size, 1) to allow broadcasting with image channels
    return window_2d[..., np.newaxis]

def apply_filter_tiled(model, image_path, patch_size=256, overlap=32):
    """
    Applies a model to a high-resolution image using overlapping patches
    and a pyramid blending window to eliminate seams.
    """
    # 1. Load the full-resolution image
    try:
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        original_img = tf.cast(img_tensor, tf.float32).numpy()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    orig_height, orig_width, _ = original_img.shape
    step = patch_size - overlap

    # Calculate and apply padding
    pad_h = (step - (orig_height - patch_size) % step) % step
    pad_w = (step - (orig_width - patch_size) % step) % step
    padded_img = cv2.copyMakeBorder(original_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    
    padded_height, padded_width, _ = padded_img.shape

    # Create empty canvases
    final_image = np.zeros(padded_img.shape, dtype=np.float32)
    weight_mask = np.zeros(padded_img.shape, dtype=np.float32)

    # --- UPDATED: Use the pyramid window for blending ---
    patch_window = _pyramid_window(patch_size)

    # 4. Iterate over the PADDED image
    print("Processing image with tiled inference (smooth blending)...")
    for y in tqdm(range(0, padded_height - patch_size + 1, step)):
        for x in range(0, padded_width - patch_size + 1, step):
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            patch_normalized = (patch / 127.5) - 1.0
            input_tensor = tf.expand_dims(patch_normalized, 0)
            predicted_patch_normalized = model.predict(input_tensor, verbose=0)[0]
            predicted_patch = (predicted_patch_normalized * 0.5 + 0.5)
            
            # Add the processed patch to our canvas, weighted by the pyramid window
            final_image[y:y+patch_size, x:x+patch_size, :] += predicted_patch * patch_window
            weight_mask[y:y+patch_size, x:x+patch_size, :] += patch_window

    # Avoid division by zero
    weight_mask[weight_mask == 0] = 1.0

    # Average the overlapping regions
    reconstructed_padded_image = (final_image / weight_mask) * 255.0
    
    # Crop back to original dimensions
    reconstructed_image = reconstructed_padded_image[0:orig_height, 0:orig_width, :]
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    print("Tiled inference complete.")
    return reconstructed_image


# This function is for the faster, lower-quality resize method
def preprocess_image(image_path, target_height, target_width):
    """
    Loads, preprocesses, and resizes an image for the simple inference method.
    """
    try:
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        orig_h = tf.shape(img_tensor)[0]
        orig_w = tf.shape(img_tensor)[1]
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None

    img_tensor = tf.cast(img_tensor, tf.float32)
    img_resized = tf.image.resize(img_tensor, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR)
    img_normalized = (img_resized / 127.5) - 1.0
    input_tensor = tf.expand_dims(img_normalized, 0)
    return input_tensor, orig_h, orig_w

# This function is for postprocessing the output of the resize method
def postprocess_output(prediction, original_height, original_width):
    """
    Denormalizes and resizes the model output back to the original image dimensions.
    """
    prediction_squeezed = prediction[0]
    prediction_denormalized = (prediction_squeezed * 0.5 + 0.5)
    prediction_resized = tf.image.resize(prediction_denormalized, [original_height, original_width], method=tf.image.ResizeMethod.BILINEAR)
    final_image = tf.clip_by_value(prediction_resized, 0.0, 1.0)
    final_image_numpy = (final_image * 255).numpy().astype(np.uint8)
    return final_image_numpy

def main():
    parser = argparse.ArgumentParser(description="Apply a trained filter to an input image.")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path where the output image will be saved.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained Keras model file (.keras or .h5).")
    # --- ADDED: Command-line flag for tiled inference ---
    parser.add_argument("-t", "--tiled", action="store_true", help="Use tiled inference for higher quality on large images.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at {args.input_path}")
        return

    start_time = time.time()

    print(f"Loading model from: {args.model_path}")
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    try:
        model_height = model.input_shape[1]
        model_width = model.input_shape[2]
        if not model_height or not model_width: raise ValueError("Model input shape not fully defined.")
        print(f"Detected model input size: {model_height}x{model_width}")
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Error: Could not automatically determine model input size. {e}")
        return

    # --- ADDED: Logic to choose inference method based on the -t flag ---
    if args.tiled:
        # Use the high-quality tiled inference method
        final_image_numpy = apply_filter_tiled(model, args.input_path, patch_size=model_height)
        if final_image_numpy is None: return # Exit if error occurred
    else:
        # Use the faster, lower-quality resize method
        print("Preprocessing input image (resize method)...")
        input_tensor, orig_h, orig_w = preprocess_image(args.input_path, model_height, model_width)
        if input_tensor is None: return

        print("Applying learned filter...")
        prediction = model.predict(input_tensor, verbose=0)

        print("Postprocessing output...")
        final_image_numpy = postprocess_output(prediction, orig_h, orig_w)
    # --- End of method selection logic ---

    process_time = time.time() - start_time

    try:
        result_image_pil = Image.fromarray(final_image_numpy)
        result_image_pil.save(args.output_path)
        print(f"\nOutput image saved successfully to: {args.output_path}")
        print(f"Total processing time: {process_time:.2f} seconds")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    main()
