import cv2
from PIL import Image, ImageTk
import numpy as np

def load_image(path):
    #load image
    return cv2.imread(path)

def save_image(path, image):
    #save image
    cv2.imwrite(path, image)

def cv_to_tk(image):
    # Convert BGR to RGB
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:  # BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # Grayscale
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Convert to Tkinter PhotoImage
    return ImageTk.PhotoImage(pil_image)

def resize_with_alpha(image, width=None, height=None):
    if image is None:
        return None
    
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    if width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        ratio = width / w
        new_w = width
        new_h = int(h * ratio)
    elif height is not None:
        ratio = height / h
        new_h = height
        new_w = int(w * ratio)
    else:
        return image
    
    # Check image
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Split into color
        b, g, r, a = cv2.split(image)
        
        # Resize each separately
        b_resized = cv2.resize(b, (new_w, new_h), interpolation=cv2.INTER_AREA)
        g_resized = cv2.resize(g, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r_resized = cv2.resize(r, (new_w, new_h), interpolation=cv2.INTER_AREA)
        a_resized = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Merge back
        resized = cv2.merge([b_resized, g_resized, r_resized, a_resized])
    else:
        # Standard resize for images 
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized