import re

def clean_ocr_text(text):
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'[^\w\s,.;:!?\'"()\-–—éèêàùçÉÈÊÀÙÇ]', ' ', text)
    text = re.sub(r'\b[A-Z0-9]{5,}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
