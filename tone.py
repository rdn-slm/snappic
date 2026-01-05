import cv2

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def black_white(image, threshold):
    gray = grayscale(image)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return bw
