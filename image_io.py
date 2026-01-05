import cv2
from PIL import Image, ImageTk

def load_image(path):
    return cv2.imread(path)

def save_image(path, image):
    cv2.imwrite(path, image)

def cv_to_tk(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(image))
