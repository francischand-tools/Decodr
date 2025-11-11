import os
import nltk
import matplotlib
from transformers import pipeline, AutoTokenizer

# ===================== CONFIGURATION =====================
os.environ['JAVA_HOME'] = r'C:\Program Files\Eclipse Adoptium\jdk-17.0.15.6-hotspot'
os.environ['PATH'] = r'C:\Program Files\Eclipse Adoptium\jdk-17.0.15.6-hotspot\bin;' + os.environ['PATH']

TESSERACT_PATH = r"C:\Users\franc\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Users\franc\AppData\Local\Programs\Tesseract-OCR\tessdata"

import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

nltk.download('punkt_tab', quiet=True)
matplotlib.use("TkAgg")

# Modèle IA
pipe = pipeline("text2text-generation", model="google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
