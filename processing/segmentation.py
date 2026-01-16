import cv2
import numpy as np
from utils.image_io import resize_with_alpha 

# Background Removal Functions
def remove_background_grabcut(image):
    if image is None:
        return None

    mask = np.zeros(image.shape[:2], np.uint8)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    height, width = image.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), 
            int(width * 0.8), int(height * 0.8))
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]
    
    # Create transparent background (RGBA)
    b_channel, g_channel, r_channel = cv2.split(foreground)
    alpha_channel = (mask2 * 255).astype(np.uint8)
    rgba_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    return rgba_image

def remove_background_simple(image, threshold=240):
    if image is None:
        return None
    
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    b, g, r = cv2.split(image)
    rgba = cv2.merge((b, g, r, mask))
    
    return rgba

def remove_background_edge(image):
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    # Fill largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Apply morphological operations to clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    b, g, r = cv2.split(image)
    rgba = cv2.merge((b, g, r, mask))
    
    return rgba

# Resizing Functions
def resize_image(image, width=None, height=None):
    return resize_with_alpha(image, width, height)

def resize_to_preset(image, preset_name):
    presets = {
        "instagram": (1080, 1080),
        "facebook": (1200, 630),
        "twitter": (1200, 675),
        "hd": (1280, 720),
        "full_hd": (1920, 1080),
        "4k": (3840, 2160)
    }
    
    if preset_name in presets:
        width, height = presets[preset_name]
        return resize_image(image, width, height)
    
    return image

# Binarization Functions
def show_binary_mask(image, threshold_method="otsu"):
    if image is None:
        return None
    
    # Handle RGBA images by converting to BGR first
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Create a copy for visualization
    if len(image.shape) == 3:
        color_img = image.copy()
    else:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding method
    if threshold_method == "otsu":
        # Automatic thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Manual threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create red overlay for foreground
    overlay = color_img.copy()
    overlay[binary == 255] = [0, 0, 255]
    
    # Blend original and overlay
    alpha = 0.5
    result = cv2.addWeighted(color_img, 1 - alpha, overlay, alpha, 0)
    
    return result

def get_binary_mask(image, threshold=127):
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:  # BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary