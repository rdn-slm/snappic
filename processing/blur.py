import cv2
import numpy as np

def gaussian_blur(image, value):
    if value == 0:
        return image.copy()
    
    # Check image
    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
    
    if has_alpha:
        b, g, r, a = cv2.split(image)
        
        # Convert value to kernel size (odd number, minimum 3)
        k = max(3, int(value / 2) * 2 + 1)
        sigma = max(0.3 * ((k - 1) * 0.5 - 1) + 0.8, 0.8)
        
        # Apply blur to color only
        b_blur = cv2.GaussianBlur(b, (k, k), sigmaX=sigma, sigmaY=sigma)
        g_blur = cv2.GaussianBlur(g, (k, k), sigmaX=sigma, sigmaY=sigma)
        r_blur = cv2.GaussianBlur(r, (k, k), sigmaX=sigma, sigmaY=sigma)
        
        # Merge back
        return cv2.merge([b_blur, g_blur, r_blur, a])
    else:
        # Convert value to kernel size (odd number, minimum 3)
        k = max(3, int(value / 2) * 2 + 1)
        
        # Apply Gaussian blur
        sigma = max(0.3 * ((k - 1) * 0.5 - 1) + 0.8, 0.8)
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

def median_blur(image, value):
    if value == 0:
        return image.copy()
    
    # Check image
    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
    
    if has_alpha:
        b, g, r, a = cv2.split(image)
        
        # Convert value to kernel size (odd number, minimum 3)
        k = max(3, int(value / 4) * 2 + 1)
        
        # Apply blur to color
        b_blur = cv2.medianBlur(b, k)
        g_blur = cv2.medianBlur(g, k)
        r_blur = cv2.medianBlur(r, k)
        
        # Merge back
        return cv2.merge([b_blur, g_blur, r_blur, a])
    else:
        # Convert value to kernel size (odd number, minimum 3)
        k = max(3, int(value / 4) * 2 + 1)
        
        return cv2.medianBlur(image, k)
