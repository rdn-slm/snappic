import cv2
import numpy as np

#Darkening
def adjust_darken(image, value):
    if value == 0:
        return image.copy()
    
    # Scale value 
    scaled_value = int(value * 2.55)
    if len(image.shape) == 3 and image.shape[2] == 4:

        b, g, r, a = cv2.split(image)
        
        # Apply darken
        b_dark = cv2.subtract(b, scaled_value)
        g_dark = cv2.subtract(g, scaled_value)
        r_dark = cv2.subtract(r, scaled_value)
        
        # Merge back
        return cv2.merge([b_dark, g_dark, r_dark, a])
    else:
        # Subtract value from image
        return cv2.subtract(image, scaled_value)

#Brightening
def adjust_brighten(image, value):
    if value == 0:
        return image.copy()
    
    # Scale value 
    scaled_value = int(value * 2.55)
    
    # Check image
    if len(image.shape) == 3 and image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        
        # Apply brighten
        b_bright = cv2.add(b, scaled_value)
        g_bright = cv2.add(g, scaled_value)
        r_bright = cv2.add(r, scaled_value)
        
        # Merge back
        return cv2.merge([b_bright, g_bright, r_bright, a])
    else:
        # Add value to image
        return cv2.add(image, scaled_value)
