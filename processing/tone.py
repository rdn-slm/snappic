import cv2
import numpy as np

#convert to grayscale
def grayscale(image):
    # Check image
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        b, g, r, a = cv2.split(image)
        
        # Merge grayscale 
        return cv2.merge([gray, gray, gray, a])
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#convert to b&w
def black_white(image, threshold):
    # Check image
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        b, g, r, a = cv2.split(image)
        
        # Apply threshold
        _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Merge B&W
        return cv2.merge([bw, bw, bw, a])
    else:
        # Convert to grayscale first 
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return bw
