import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract

def deskew_image(pil_image):
    img = np.array(pil_image.convert("L"))
    img = cv2.bitwise_not(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if abs(angle) == 90:
        angle = 0
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    result = Image.fromarray(cv2.bitwise_not(rotated))
    result = result.convert("L")
    result = ImageEnhance.Contrast(result).enhance(2.0)
    result = ImageEnhance.Sharpness(result).enhance(2.0)
    return result

def correct_orientation(pil_image):
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        osd = pytesseract.image_to_osd(img)
        for line in osd.split("\n"):
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
                if angle != 0:
                    return pil_image.rotate(-angle, expand=True)
    except Exception:
        pass
    return pil_image

def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def enhance_sharpness(img):
    kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def remove_noise(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return cv2.medianBlur(img, 3)

def adjust_brightness_contrast(img, alpha=1.3, beta=15):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def remove_borders(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(sorted(contours, key=cv2.contourArea, reverse=True)[0])
        return img[y:y+h, x:x+w]
    return img
