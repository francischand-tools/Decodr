import os
import sys
import re
import textwrap
import torch
import nltk
import cv2
import numpy as np
import pytesseract
import fitz
import docx2txt
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import language_tool_python
import matplotlib
import tkinter
import tkinter.messagebox
import customtkinter

from PIL import Image, ImageTk, ImageQt, ImageEnhance
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pytesseract import Output
from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config.settings import tokenizer
from ocr.orientation import correct_orientation, deskew_image
from ocr.preprocessing import preprocess_image
from ocr.text_cleaning import clean_ocr_text


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        customtkinter.set_appearance_mode("Light")
        customtkinter.set_default_color_theme("modern_dark.json")

        # configure window
        self.title("Decodr - Image analysis")
        self.geometry(f"{1100}x{580}")
        self.attributes("-fullscreen", False)
        self.resizable(False, False)

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # sidebar (idem ton code)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Decodr", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Upload an image", command=self.open_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Upload a manuscript", command=self.show_notification)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # Boutons Analysis + Restart OCR (idem)
        self.button_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=1, column=1, padx=(20, 0), pady=(0, 0), sticky="n")
        largeur_totale = 240
        largeur_bouton = largeur_totale // 2
        self.main_button_1 = customtkinter.CTkButton(
            master=self.button_frame,
            fg_color="transparent",
            text="Analysis",
            border_width=2,
            width=largeur_bouton,
            text_color=("gray10", "#4A536B"),
            command=self.run_analysis
        )
        self.main_button_1.grid(row=0, column=0, padx=(0, 10))
        self.main_button_2 = customtkinter.CTkButton(
            master=self.button_frame,
            fg_color="transparent",
            text="Restart OCR",
            border_width=2,
            width=largeur_bouton,
            text_color=("gray10", "#4A536B"),
            command=self.reset_ocr
        )
        self.main_button_2.grid(row=0, column=1)

        # Configure la grille principale
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(1, weight=1) 

        # Crée le conteneur central
        self.center_container = customtkinter.CTkFrame(self, fg_color="#47506B", corner_radius=0)
        self.center_container.grid(row=0, column=1, padx=5, pady=5) # "sticky="nsew"

        # Configure la grille du conteneur central pour accueillir canvas + scrollbars
        self.center_container.grid_rowconfigure(0, weight=1)
        self.center_container.grid_columnconfigure(0, weight=1)

        # Canvas dans le container
        self.center_canvas = tk.Canvas(self.center_container, bg="#1A1A1A", highlightthickness=0)
        self.center_canvas.config(width=600, height=310)
        self.center_canvas.grid(row=0, column=0) # "sticky="nsew"

        # Scrollbars dans le container
        self.scrollbar_y = ttk.Scrollbar(self.center_container, orient="vertical", command=self.center_canvas.yview)
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")

        self.scrollbar_x = ttk.Scrollbar(self.center_container, orient="horizontal", command=self.center_canvas.xview)
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")

        # Configure le canvas pour utiliser les scrollbars
        self.center_canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        # Frame qui contiendra l’image dans le canvas
        self.image_frame = customtkinter.CTkFrame(self.center_canvas, fg_color="#1A1A1A", corner_radius=0)
        self.canvas_window = self.center_canvas.create_window((0, 0), window=self.image_frame, anchor="nw")

        # Bind pour ajuster la scrollregion
        self.image_frame.bind("<Configure>", self.update_scrollregion)
        self.center_canvas.bind("<Configure>", self.resize_canvas)

        # Label dans image_frame
        self.image_label = tk.Label(self.image_frame, bg="#1A1A1A", text="Drop or open an image", fg="white")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Configuration poids dans image_frame
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        # Configure la grille principale pour la ligne 2 aussi
        self.grid_rowconfigure(2, weight=0)  # hauteur fixe pour la zone texte
        self.grid_columnconfigure(1, weight=1)  # même colonne 1 que canvas/image

        # Crée un conteneur pour la zone texte (sous image et boutons)
        self.text_frame = customtkinter.CTkFrame(self, width=250, height=150, corner_radius=10, border_width=2, border_color="gray")
        self.text_frame.grid(row=2, column=1, padx=(20, 0), pady=(10, 0), sticky="nsew")

        # Configure la grille interne du conteneur text_frame
        self.text_frame.grid_rowconfigure(0, weight=1)
        self.text_frame.grid_columnconfigure(0, weight=1)

        # Text widget pour OCR
        self.text_box = tkinter.Text(self.text_frame, wrap="word", height=8)
        self.text_box.grid(row=0, column=0, sticky="nsew")

        # Scrollbar verticale pour la text_box
        self.text_scrollbar = tkinter.Scrollbar(self.text_frame, command=self.text_box.yview)
        self.text_scrollbar.grid(row=0, column=1, sticky="ns")

        # Lier la scrollbar au text_box
        self.text_box.config(yscrollcommand=self.text_scrollbar.set)

        # Tabview (idem)
        self.tabview = customtkinter.CTkTabview(self, width=250, height=180)
        self.tabview.grid(row=0, column=2, padx=(20, 10), pady=(20, 10), sticky="n")
        self.tabview.add("General")
        self.tabview.tab("General").grid_columnconfigure(0, weight=1)

        self.combobox_1 = customtkinter.CTkComboBox(self.tabview.tab("General"),
                                                    values=["English", "French", "Chinese"])
        self.combobox_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.combobox_2 = customtkinter.CTkComboBox(self.tabview.tab("General"),
                                                    values=[".txt", ".jpeg", ".jpg"])
        self.combobox_2.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.save_button = customtkinter.CTkButton(self.tabview.tab("General"),
            text="Save As...",
            command=lambda: self.open_file_explorer()  # Récupère le texte OCR
        )
        self.save_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Bottom frame = graphique matplotlib
        self.bottom_frame = customtkinter.CTkFrame(self, height=180, corner_radius=10, fg_color="transparent",
                                                  border_width=1, border_color="gray")
        self.bottom_frame.grid(row=2, column=2, padx=(20, 10), pady=(0, 10), sticky="nsew")

        self.figure = plt.Figure(figsize=(3, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.bottom_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")

        self.draw_percent_pie(0)  # Init à 0%

        # Variables
        self.loaded_image = None  # stocker l'image PIL pour analyse

        # set defaults
        self.appearance_mode_optionemenu.set("Light")
        self.scaling_optionemenu.set("100%")
        self.combobox_1.set("English")
        self.combobox_2.set(".png")

    def update_scrollregion(self, event=None):
        self.center_canvas.configure(scrollregion=self.center_canvas.bbox("all"))
        self.toggle_scrollbars()

    def resize_canvas(self, event=None):
        self.toggle_scrollbars()

    def toggle_scrollbars(self):
        bbox = self.center_canvas.bbox("all")
        if not bbox:
            return

        canvas_width = self.center_canvas.winfo_width()
        canvas_height = self.center_canvas.winfo_height()
        content_width = bbox[2] - bbox[0]
        content_height = bbox[3] - bbox[1]

        # Afficher/masquer la scrollbar verticale
        if content_height > canvas_height:
            self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        else:
            self.scrollbar_y.grid_forget()

        # Afficher/masquer la scrollbar horizontale
        if content_width > canvas_width:
            self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        else:
            self.scrollbar_x.grid_forget()

    def draw_percent_pie(self, percent):
        self.ax.clear()
        sizes = [percent, 100 - percent]
        colors = ['#4CAF50', '#E0E0E0']  # Vert et gris clair

        wedges, _ = self.ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3)
        )

        self.ax.text(0, 0, f"{percent:.0f}%", ha='center', va='center', fontsize=16, fontweight='bold')
        self.ax.set_aspect('equal')
        self.canvas.draw()

    def run_analysis(self):
        if self.loaded_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        # 1️⃣ Correction d’orientation et redressement
        oriented = correct_orientation(self.loaded_image)
        deskewed = deskew_image(oriented)

        # 2️⃣ Sélection de la langue et du mode
        selected_lang = self.combobox_1.get()

        if selected_lang == "English":
            lang = "eng"
            mode = "print"
        elif selected_lang == "French":
            lang = "fra"
            mode = "print"
        elif selected_lang == "Chinese":
            lang = "chi_sim"
            mode = "chinese"
        else:
            lang = "eng"
            mode = "print"

        # 3️⃣ Prétraitement de l’image selon la langue
        processed = preprocess_image(oriented, mode=mode)
        processed.info['dpi'] = (300, 300)

        # 4️⃣ Configuration OCR
        config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

        # 5️⃣ OCR complet
        data = pytesseract.image_to_data(processed, lang=lang, config=config, output_type=pytesseract.Output.DICT)

        ocr_text = " ".join([
            data['text'][i]
            for i in range(len(data['text']))
            if data['text'][i].strip() != ""
        ])
        print("Texte OCR avant correction (aperçu 500 caractères max) :")
        print(ocr_text[:500])
        print("=" * 80)

        tokens = tokenizer.encode(ocr_text, return_tensors="pt")
        print(f"Nombre de tokens : {tokens.shape[1]}")

        # 🧹 Nettoyage du texte OCR
        clean_text = clean_ocr_text(ocr_text)

        # 6️⃣ Affiche le texte OCR nettoyé
        print("Texte brut OCR nettoyé :\n", clean_text)
        self.text_box.delete("1.0", tkinter.END)
        self.text_box.insert(tk.END, clean_text)

        # 7️⃣ Affiche les boîtes de détection
        self.display_boxes_on_image(processed, lang=lang)




    

    def display_boxes_on_image(self, pil_image, lang="eng"):
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        data = pytesseract.image_to_data(img_cv, lang=lang, output_type=Output.DICT)
        print("Mots détectés par image_to_data:", [
            data['text'][i]
            for i in range(len(data['text']))
            if data['text'][i].strip() != "" and int(data['conf'][i]) > 0
        ])


        words = []
        confidences = []
        color = (0, 255, 0)  # vert

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word_text = data['text'][i].strip()
            conf_str = data['conf'][i]
            try:
                conf = float(conf_str)
            except:
                conf = -1

            if word_text != '':
                words.append(word_text)
                if conf >= 0:
                    confidences.append(conf)

            if int(data["conf"][i]) > 0:
                (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_cv, word_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        avg_conf = int(sum(confidences) / len(confidences)) if confidences else 0
        self.draw_percent_pie(avg_conf)

        pil_with_boxes = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil_with_boxes)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    
    def show_notification(self):
        messagebox.showinfo("Notification", "This feature is not supported in this version and is currently in the process of being finalized.")


    def open_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Images & documents", "*.png *.jpg *.jpeg *.bmp *.tiff *.pdf *.docx *.doc")]
        )
        if not filepath:
            return
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                image = Image.open(filepath)
                self.loaded_image = image
                self.display_image(image)

            elif ext == ".pdf":
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=300)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                self.loaded_image = image
                self.display_image(image)

            elif ext in [".docx", ".doc"]:
                import docx2txt
                text = docx2txt.process(filepath)
                self.text_box.delete("1.0", tkinter.END)
                self.text_box.insert(tkinter.END, text)
                self.loaded_image = None  # Pas d’image à traiter
            else:
                messagebox.showerror("Erreur", "Type de fichier non supporté.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement : {e}")
 
    def display_image(self, image):
        try:
            self.current_image = image  # pour traitements ultérieurs

            # Supprime l'image précédente si elle existe
            if hasattr(self, 'image_label'):
                self.image_label.destroy()

            # Convertit l'image pour tkinter
            tk_image = ImageTk.PhotoImage(image)
            self.tk_image = tk_image  # garder une référence

            # Affiche dans image_frame (qui est dans center_canvas)
            self.image_label = customtkinter.CTkLabel(self.image_frame, image=tk_image, text="")
            self.image_label.pack()

            # Met à jour la scrollregion et active les scrollbars si nécessaire
            self.center_canvas.update_idletasks()
            self.center_canvas.configure(scrollregion=self.center_canvas.bbox("all"))
            self.toggle_scrollbars()

        except Exception as e:
            print(f"Failed to display image: {e}")


    def reset_ocr(self):
        self.text_box.delete("1.0", tkinter.END)
        self.image_label.configure(image=None, text="No image")
        self.image_label.image = None
        self.draw_percent_pie(0)
        self.loaded_image = None

    def open_file_explorer(self):
        selected_language = self.combobox_1.get()
        selected_format = self.combobox_2.get()

        # Récupérer le texte OCR directement depuis la zone texte
        ocr_text = self.text_box.get("1.0", "end-1c")

        # Traduction
        if selected_language == "French":
            traduction = GoogleTranslator(source='auto', target='fr').translate(ocr_text)
        elif selected_language == "Chinese":
            traduction = GoogleTranslator(source='auto', target='zh-CN').translate(ocr_text)
        else:
            traduction = GoogleTranslator(source='auto', target='en').translate(ocr_text)

        # Ouvre l'explorateur de fichiers
        filetypes = [(f"{selected_format.upper()} files", f"*{selected_format}"), ("All files", "*.*")]
        filepath = filedialog.asksaveasfilename(defaultextension=selected_format, filetypes=filetypes)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(traduction)


    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
