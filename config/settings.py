import os
import nltk
import matplotlib
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv

load_dotenv() 

TESSERACT_PATH = os.environ.get("TESSERACT_PATH")
JAVA_HOME = os.environ.get("JAVA_HOME")



import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

nltk.download('punkt_tab', quiet=True)
matplotlib.use("TkAgg")

# Modèle IA
pipe = pipeline("text2text-generation", model="google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
