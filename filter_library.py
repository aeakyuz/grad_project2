import cv2
import numpy as np

def apply_oil_painting_filter(img, size=10, dyn_ratio=1):
    """Applies the oil painting filter to a single image."""
    return cv2.xphoto.oilPainting(img, size, dyn_ratio)

def apply_pencil_sketch_filter(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
    """
    Applies the pencil sketch filter to a single image.
    
    CORRECTED: This now returns a single-channel grayscale image to match
    the training data generation process.
    """
    _ , sketch_gray = cv2.pencilSketch(
        img,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        shade_factor=shade_factor
    )
    # Return the single-channel grayscale image directly
    return sketch_gray

def apply_grayscale_filter(img):
    """Applies a grayscale filter to a single image."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_sepia_filter(img):
    """Applies a sepia tone filter to a single image."""
    img_float = img.astype(np.float32)
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img_float = cv2.transform(img_float, sepia_matrix)
    sepia_img = np.clip(sepia_img_float, 0, 255).astype(np.uint8)
    return sepia_img

def apply_cartoon_effect(img, d=9, sigma_color=250, sigma_space=250, block_size=9, c=2):
    """Applies a cartoon/posterization effect to a single image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=block_size,
                                  C=c)
    color = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_img = cv2.bitwise_and(color, edges_color)
    return cartoon_img

def apply_gaussian_blur(img, kernel_size_val=31):
    """Applies a Gaussian blur filter to a single image."""
    # Kernel size must be an odd positive number
    if kernel_size_val % 2 == 0:
        kernel_size_val += 1
    return cv2.GaussianBlur(img, (kernel_size_val, kernel_size_val), 0)

# A dictionary to easily access the functions by name
FILTER_FUNCTIONS = {
    "oil": apply_oil_painting_filter,
    "sketch": apply_pencil_sketch_filter,
    "grayscale": apply_grayscale_filter,
    "sepia": apply_sepia_filter,
    "cartoon": apply_cartoon_effect,
    "gaussian": apply_gaussian_blur,
}

