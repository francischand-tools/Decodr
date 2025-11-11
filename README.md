# Decodr

Decodr is a desktop application for analyzing images and extracting text from manuscripts using OCR. It supports multiple languages and provides features like text correction, translation, and confidence visualization.

<img width="1097" height="609" alt="image" src="https://github.com/user-attachments/assets/9ed3d6a7-3c6f-4e93-9b21-8e05d77068e5" />

## Features

- Drag and drop images or upload files
- OCR text extraction (English, French, Chinese)
- Text correction and preprocessing
- Translation support
- Confidence percentage visualization
- Save OCR results in multiple formats (.txt, .jpeg, .jpg)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/francischand-tools/Decodr.git
   cd Decodr

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux

3. Install dependencies:
     ```bash
     pip install -r requirements.txt

4. Create a .env file with your local paths:
    ```bash
    TESSERACT_PATH=C:\path\to\tesseract.exe
    JAVA_HOME=C:\path\to\jdk
   
5. Run the application:
    ```bash
    python main.py

## Usage

- Drag and drop images or upload files
- Select the language(s) used in the image
- Click on "Analyze"
- Choose the target language and file format to translate and save the text

## License

This project is licensed under the MIT License.


