import tkinter as tk
from tkinter import ttk, filedialog
from utils.image_io import load_image, save_image, cv_to_tk
from processing.blur import gaussian_blur, median_blur
from processing.tone import grayscale, black_white
from processing.light import adjust_brightness

class SnappicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SNAPPIC")
        self.geometry("1100x600")
        self.configure(bg="#1e1e1e")

        self.original = None
        self.processed = None

        self.create_layout()

    def create_layout(self):
        # Left Panel (History)
        self.history = tk.Listbox(self, width=25, bg="#2a2a2a", fg="white")
        self.history.pack(side="left", fill="y", padx=10)

        # Center Image Panel
        self.image_label = tk.Label(self, bg="#1e1e1e")
        self.image_label.pack(side="left", expand=True)

        # Right Control Panel
        control = ttk.Notebook(self)
        control.pack(side="right", fill="y", padx=10)

        self.blur_tab(control)
        self.color_tab(control)
        self.light_tab(control)

        # Bottom buttons
        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.pack(side="bottom", pady=10)

        tk.Button(btn_frame, text="LOAD NEW", command=self.load).pack(side="left", padx=5)
        tk.Button(btn_frame, text="SAVE", command=self.save).pack(side="left", padx=5)

    def blur_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="BLUR")

        tk.Scale(tab, from_=0, to=100, label="Gaussian Blur",
                 command=self.apply_gaussian).pack(fill="x", pady=10)

        tk.Scale(tab, from_=0, to=100, label="Median Blur",
                 command=self.apply_median).pack(fill="x")

    def color_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="COLOR")

        tk.Button(tab, text="GRAYSCALE", command=self.apply_gray).pack(fill="x", pady=10)
        tk.Scale(tab, from_=0, to=255, label="Black & White",
                 command=self.apply_bw).pack(fill="x")

    def light_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="LIGHT")

        tk.Scale(tab, from_=-100, to=100, label="Brightness",
                 command=self.apply_light).pack(fill="x", pady=10)

    def load(self):
        path = filedialog.askopenfilename()
        self.original = load_image(path)
        self.processed = self.original.copy()
        self.update_image(self.original)
        self.history.insert("end", "Before Image")

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        save_image(path, self.processed)

    def update_image(self, img):
        self.tk_img = cv_to_tk(img)
        self.image_label.config(image=self.tk_img)

    # Processing callbacks
    def apply_gaussian(self, v):
        self.processed = gaussian_blur(self.original, int(v))
        self.update_image(self.processed)
        self.history.insert("end", "Gaussian Blur")

    def apply_median(self, v):
        self.processed = median_blur(self.original, int(v))
        self.update_image(self.processed)
        self.history.insert("end", "Median Blur")

    def apply_gray(self):
        self.processed = grayscale(self.original)
        self.update_image(self.processed)
        self.history.insert("end", "Grayscale")

    def apply_bw(self, v):
        self.processed = black_white(self.original, int(v))
        self.update_image(self.processed)
        self.history.insert("end", "Black & White")

    def apply_light(self, v):
        self.processed = adjust_brightness(self.original, int(v))
        self.update_image(self.processed)
        self.history.insert("end", "Brightness Adjusted")
