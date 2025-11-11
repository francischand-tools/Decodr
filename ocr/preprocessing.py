import cv2
import numpy as np
from PIL import Image, ImageEnhance
from .orientation import deskew_image
from .orientation import remove_borders, remove_noise, adjust_brightness_contrast, increase_contrast, enhance_sharpness, unsharp_mask

def preprocess_image(pil_image, mode="print"):
    pil_image = deskew_image(pil_image)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = increase_contrast(img)
    img = enhance_sharpness(img)
    img = unsharp_mask(img)
    img = adjust_brightness_contrast(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = remove_noise(gray)

    if mode == "handwritten":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif mode == "chinese":
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

    binarized = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 9)
    binarized = remove_borders(binarized)
    resized = cv2.resize(binarized, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    final_img = Image.fromarray(resized)
    final_img.info["dpi"] = (300, 300)
    return final_img
