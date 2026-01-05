import cv2

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)
