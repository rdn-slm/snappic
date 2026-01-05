import cv2

def gaussian_blur(image, value):
    k = max(1, value // 2 * 2 + 1)
    return cv2.GaussianBlur(image, (k, k), 0)

def median_blur(image, value):
    k = max(1, value // 2 * 2 + 1)
    return cv2.medianBlur(image, k)
