import tkinter as tk
import numpy as np
import cv2
from tkinter import ttk, filedialog, messagebox
from utils.image_io import load_image, save_image, cv_to_tk
from processing.blur import gaussian_blur, median_blur
from processing.tone import grayscale, black_white
from processing.light import adjust_darken, adjust_brighten
from processing.segmentation import (
    remove_background_grabcut, remove_background_simple, remove_background_edge,
    resize_image, resize_to_preset, show_binary_mask, get_binary_mask
)

# ---- DEFINE THE PARAMETERS ----
class SnappicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Values for window styling
        self.title("SNAPPIC - Photo Editor")
        self.geometry("1200x700")
        self.configure(bg="#1e1e1e")
        
        self.minsize(1000, 600)

        # Initializing pic for backup 
        self.original = None
        self.processed = None
        
        # Filter values
        self.gaussian_value = 0
        self.median_value = 0
        self.darken_value = 0 
        self.brighten_value = 0  
        self.is_grayscale = False
        self.is_blackwhite = False
        self.bw_threshold = 127
        self.bg_threshold = 240  
        self.show_binary = False
        
        # Track background removal state
        self.has_background_removed = False
        self.background_method = None  
        
        # Track selective blurring state
        self.selective_blur_mode = False
        self.current_mask = None
        self.mask_start = None
        self.mask_points = []
        self.current_mask_type = "rectangle"
        self.selective_intensity = 50
        self.selective_blur_type = "gaussian"
        self.mask_history = []  # List of (mask, blur_type, intensity)
        
        # Track if median blur has been used for user feedback
        self.median_blur_used = False  
        
        # Keep uncropped original (to revert back to uncropped version)
        self.original_backup = None  

        # Cropping features
        self.crop_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        
        self.create_layout()
        self.bind_mouse_events()

    def bind_mouse_events(self):
        # Bind mouse events from user for selective blur drawing and cropping features
        if hasattr(self, 'image_label'):
            # For selective blur
            self.image_label.bind("<Button-1>", self.start_mask_draw)
            self.image_label.bind("<B1-Motion>", self.draw_mask)
            self.image_label.bind("<ButtonRelease-1>", self.finish_mask_draw)
            
            # For cropping (with different mouse buttons or modifiers)
            self.image_label.bind("<Control-Button-1>", self.start_crop)  # Ctrl+Click for crop
            self.image_label.bind("<Control-B1-Motion>", self.draw_crop)
            self.image_label.bind("<Control-ButtonRelease-1>", self.finish_crop)
            
# ---- CREATE LAYOUT INTERFACE -----
    def create_layout(self):
        # Main container 
        main_container = tk.Frame(self, bg="#1e1e1e")
        main_container.pack(fill="both", expand=True)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=3)
        main_container.grid_columnconfigure(2, weight=0)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=0)

        # Panel History
        history_frame = tk.Frame(main_container, bg="#1e1e1e")
        history_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        tk.Label(history_frame, text="HISTORY", font=("Arial", 12, "bold"), 
                fg="white", bg="#1e1e1e").pack()
        self.history = tk.Listbox(history_frame, width=25, bg="#2a2a2a", fg="white")
        self.history.pack(fill="both", expand=True, pady=5)

        # Panel Image Display
        image_frame = tk.Frame(main_container, bg="#1e1e1e")
        image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.image_label = tk.Label(image_frame, bg="#1e1e1e")
        self.image_label.pack(fill="both", expand=True)
        
        self.placeholder_label = tk.Label(image_frame, 
                                        text="Load an image to begin editing",
                                        fg="#888888", bg="#1e1e1e",
                                        font=("Arial", 14))
        self.placeholder_label.pack(fill="both", expand=True)

        # Panel Controls
        control_frame = tk.Frame(main_container, bg="#1e1e1e", width=320)
        control_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        control_frame.grid_propagate(False)
        
        control = ttk.Notebook(control_frame, width=300)
        control.pack(fill="both", expand=True)

        # Tab controls
        self.blur_tab(control)
        self.color_tab(control)
        self.light_tab(control)
        self.segmentation_tab(control)
        self.resize_tab(control)

        # Configure button for I/O process and reset filter
        btn_frame = tk.Frame(main_container, bg="#1e1e1e")
        btn_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        button_container = tk.Frame(btn_frame, bg="#1e1e1e")
        button_container.pack()
        
        tk.Button(button_container, text="OPEN", command=self.load, 
                width=10, height=1).pack(side="left", padx=5)
        tk.Button(button_container, text="SAVE", command=self.save,
                width=10, height=1).pack(side="left", padx=5)
        tk.Button(button_container, text="RESET ALL", command=self.reset_filters,
                width=10, height=1).pack(side="left", padx=5)

        # Menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.load)
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        

# ---- CREATE BLURRING TAB -----
    def blur_tab(self, notebook):
        tab = tk.Frame(notebook, bg="#1e1e1e")
        notebook.add(tab, text="BLUR")
        
        tk.Label(tab, text="Blur Filters", font=("Arial", 12, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        # Global Blur Section for both Gaussian and Median
        global_frame = tk.LabelFrame(tab, text="Global Blur", 
                                    font=("Arial", 10),
                                    fg="white", bg="#1e1e1e",
                                    relief="ridge")
        global_frame.pack(fill="x", padx=10, pady=5)
        
        # Gaussian Blur
        tk.Label(global_frame, text="Gaussian Blur", font=("Arial", 9),
                fg="white", bg="#1e1e1e").pack(pady=(5, 0))
        self.gaussian_slider = tk.Scale(global_frame, from_=0, to=100, orient="horizontal",
                                    length=250, showvalue=True, 
                                    bg="#1e1e1e", fg="white",
                                    troughcolor="#333333",
                                    highlightbackground="#1e1e1e",
                                    command=self.apply_gaussian)
        self.gaussian_slider.pack(pady=5, padx=10)
        
        # Median Blur
        tk.Label(global_frame, text="Median Blur", font=("Arial", 9),
                fg="white", bg="#1e1e1e").pack(pady=(15, 0))
        self.median_slider = tk.Scale(global_frame, from_=0, to=100, orient="horizontal",
                                    length=250, showvalue=True,
                                    bg="#1e1e1e", fg="white",
                                    troughcolor="#333333",
                                    highlightbackground="#1e1e1e",
                                    command=self.apply_median)
        self.median_slider.pack(pady=5, padx=10)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Selective Blur Section
        selective_frame = tk.LabelFrame(tab, text="Selective Blur", 
                                    font=("Arial", 10),
                                    fg="white", bg="#1e1e1e",
                                    relief="ridge")
        selective_frame.pack(fill="x", padx=10, pady=5)
        
        # Toggle button for Selective Blur
        self.selective_toggle_btn = tk.Button(selective_frame, 
                                            text="ENABLE SELECTIVE MODE",
                                            command=self.toggle_selective_mode,
                                            font=("Arial", 9),
                                            width=20, height=1,
                                            bg="#2a2a2a", fg="white")
        self.selective_toggle_btn.pack(pady=10)
        
        # Shape selection
        shape_frame = tk.Frame(selective_frame, bg="#1e1e1e")
        shape_frame.pack(pady=5)
        
        tk.Label(shape_frame, text="Shape:", fg="white", bg="#1e1e1e").pack(side="left", padx=5)
        
        # Initialize the shape with rectangle, then user can choose to go for other shapes
        # Selective Blurring
        self.mask_shape_var = tk.StringVar(value="rectangle")
        tk.Radiobutton(shape_frame, text="Rectangle", variable=self.mask_shape_var,
                    value="rectangle", fg="white", bg="#1e1e1e",
                    selectcolor="#2a2a2a",
                    command=lambda: setattr(self, 'current_mask_type', 'rectangle')).pack(side="left", padx=5)
        tk.Radiobutton(shape_frame, text="Circle", variable=self.mask_shape_var,
                    value="circle", fg="white", bg="#1e1e1e",
                    selectcolor="#2a2a2a",
                    command=lambda: setattr(self, 'current_mask_type', 'circle')).pack(side="left", padx=5)
        tk.Radiobutton(shape_frame, text="Freeform", variable=self.mask_shape_var,
                    value="freeform", fg="white", bg="#1e1e1e",
                    selectcolor="#2a2a2a",
                    command=lambda: setattr(self, 'current_mask_type', 'freeform')).pack(side="left", padx=5)
        
        # Blur type selection: Gaussian Blur and Median Blur
        # Note that Median Blurring is computationally expensive, so only one-time is allowed
        type_frame = tk.Frame(selective_frame, bg="#1e1e1e")
        type_frame.pack(pady=5)
        
        tk.Label(type_frame, text="Blur Type:", fg="white", bg="#1e1e1e").pack(side="left", padx=5)
        
        self.selective_blur_var = tk.StringVar(value="gaussian")
        tk.Radiobutton(type_frame, text="Gaussian", variable=self.selective_blur_var,
                    value="gaussian", fg="white", bg="#1e1e1e",
                    selectcolor="#2a2a2a",
                    command=lambda: setattr(self, 'selective_blur_type', 'gaussian')).pack(side="left", padx=5)
        tk.Radiobutton(type_frame, text="Median", variable=self.selective_blur_var,
                    value="median", fg="white", bg="#1e1e1e",
                    selectcolor="#2a2a2a",
                    command=lambda: setattr(self, 'selective_blur_type', 'median')).pack(side="left", padx=5)
        
        # Intensity slider
        tk.Label(selective_frame, text="Intensity:", fg="white", bg="#1e1e1e").pack(pady=(5,0))
        self.selective_intensity_slider = tk.Scale(selective_frame, from_=0, to=100, orient="horizontal",
                                                length=250, showvalue=True,
                                                bg="#1e1e1e", fg="white",
                                                troughcolor="#333333",
                                                highlightbackground="#1e1e1e",
                                                command=lambda v: setattr(self, 'selective_intensity', int(v)))
        self.selective_intensity_slider.set(50)
        self.selective_intensity_slider.pack(pady=5, padx=10)
        
        # Control buttons
        btn_frame = tk.Frame(selective_frame, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Clear All", 
                command=self.clear_selective_areas,
                width=10, height=1,
                bg="#2a2a2a", fg="white").pack(side="left", padx=5)
        
        tk.Button(btn_frame, text="Undo Last", 
                command=self.undo_last_mask,
                width=10, height=1,
                bg="#2a2a2a", fg="white").pack(side="left", padx=5)
        
        # Instructions
        tk.Label(selective_frame, 
                text="Enable selective mode, then click & drag on image",
                fg="#888888", bg="#1e1e1e", font=("Arial", 8)).pack(pady=5)
        

# ---- CREATE COLOR/TONING TAB  -----
    def color_tab(self, notebook):
        tab = tk.Frame(notebook, bg="#1e1e1e")
        notebook.add(tab, text="COLOR")
        
        tk.Label(tab, text="Color Operations", font=("Arial", 12, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        # Grayscale section
        self.grayscale_btn = tk.Button(tab, text="GRAYSCALE", 
                                    command=self.apply_gray,
                                    font=("Arial", 10, "bold"),
                                    width=20, height=2,
                                    bg="#2a2a2a", fg="white")
        self.grayscale_btn.pack(pady=10)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=10)
        
        # Black & White section
        tk.Label(tab, text="Black & White", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=5)
        
        # B&W button
        self.bw_toggle_btn = tk.Button(tab, text="BLACK & WHITE", 
                                    command=self.toggle_blackwhite,
                                    font=("Arial", 9),
                                    width=15, height=1,
                                    bg="#2a2a2a", fg="white")
        self.bw_toggle_btn.pack(pady=5)
        
        # B&W threshold
        tk.Label(tab, text="Threshold", fg="white", bg="#1e1e1e").pack()
        self.bw_slider = tk.Scale(tab, from_=0, to=255, orient="horizontal",
                                length=250, showvalue=True,
                                bg="#1e1e1e", fg="white",
                                troughcolor="#333333",
                                highlightbackground="#1e1e1e",
                                command=self.apply_bw)
        self.bw_slider.set(127)
        self.bw_slider.config(state="disabled")
        self.bw_slider.pack(pady=5, padx=10)

# ---- CREATE LIGHT TAB -----
    def light_tab(self, notebook):
        tab = tk.Frame(notebook, bg="#1e1e1e")
        notebook.add(tab, text="LIGHT")
        
        tk.Label(tab, text="Brightness Adjustment", font=("Arial", 12, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        # Darken 
        tk.Label(tab, text="Darken", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=(5, 0))
        self.darken_slider = tk.Scale(tab, from_=0, to=100, orient="horizontal",
                                    length=250, showvalue=True,
                                    bg="#1e1e1e", fg="white",
                                    troughcolor="#333333",
                                    highlightbackground="#1e1e1e",
                                    command=self.apply_darken)
        self.darken_slider.pack(pady=5, padx=10)
        
        # Brighten 
        tk.Label(tab, text="Brighten", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=(15, 0))
        self.brighten_slider = tk.Scale(tab, from_=0, to=100, orient="horizontal",
                                    length=250, showvalue=True,
                                    bg="#1e1e1e", fg="white",
                                    troughcolor="#333333",
                                    highlightbackground="#1e1e1e",
                                    command=self.apply_brighten)
        self.brighten_slider.pack(pady=5, padx=10)

# ---- CREATE SEGMENTATION TAB -----
    def segmentation_tab(self, notebook):
        tab = tk.Frame(notebook, bg="#1e1e1e")
        notebook.add(tab, text="SEGMENTATION")
        
        tk.Label(tab, text="Background Removal", font=("Arial", 12, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        tk.Label(tab, text="Select Method:", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=5)
        
        # Background removal
        bg_frame = tk.Frame(tab, bg="#1e1e1e")
        bg_frame.pack(pady=10)
        
        # Button for GrabCut, Simple BG Removal and Edge-Based features
        tk.Button(bg_frame, text="GrabCut (Advanced)", 
                command=lambda: self.apply_background_removal("grabcut"),
                width=20, height=2,
                bg="#2a2a2a", fg="white").pack(pady=5)
        
        tk.Button(bg_frame, text="Simple (White BG)", 
                command=lambda: self.apply_background_removal("simple"),
                width=20, height=2,
                bg="#2a2a2a", fg="white").pack(pady=5)
    
        tk.Button(bg_frame, text="Edge-Based", 
                command=lambda: self.apply_background_removal("edge"),
                width=20, height=2,
                bg="#2a2a2a", fg="white").pack(pady=5)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Threshold
        tk.Label(tab, text="Background Threshold (for Simple method):", 
                fg="white", bg="#1e1e1e", font=("Arial", 9)).pack(pady=5)
        
        self.bg_threshold_slider = tk.Scale(tab, from_=200, to=255, orient="horizontal",
                                        length=250, showvalue=True,
                                        bg="#1e1e1e", fg="white",
                                        troughcolor="#333333",
                                        highlightbackground="#1e1e1e",
                                        command=self.update_bg_threshold)
        self.bg_threshold_slider.set(240)
        self.bg_threshold_slider.pack(pady=5, padx=10)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Binarization
        tk.Label(tab, text="Show Binary Mask", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=5)
        
        self.binary_toggle_btn = tk.Button(tab, text="SHOW BINARY MASK", 
                                        command=self.toggle_binary_mask,
                                        font=("Arial", 9),
                                        width=20, height=2,
                                        bg="#2a2a2a", fg="white")
        self.binary_toggle_btn.pack(pady=10)
        
        # Info
        tk.Label(tab, text="Note: Save as PNG to keep transparency", 
                fg="#888888", bg="#1e1e1e", font=("Arial", 8)).pack(pady=10)

    def reset_crop_only(self):
        # Reset crop without affecting other filters.
        print("DEBUG: reset_crop_only called")
        if self.original is not None:
            
            self.apply_all_filters()
            
            # Turn off crop mode
            if self.crop_mode:
                self.crop_mode = False
                if hasattr(self, 'crop_toggle_btn'):
                    self.crop_toggle_btn.config(text="ENABLE CROP", 
                                              bg="#2a2a2a", fg="white")
            
            # Clear crop selection
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None
            
            self.history.insert("end", "Crop reset")
        else:
            messagebox.showwarning("No Image", "No image to reset")
            

# ---- CREATE LOAD TAB -----
    def load(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            try:
                self.original = load_image(path)
                self.processed = self.original.copy()
                self.original_backup = self.original.copy()
                self.reset_filters()
                self.update_image(self.processed)
                self.placeholder_label.pack_forget()
                self.history.insert("end", f"Loaded: {path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

# ---- CREATE SAVE TAB -----
    def save(self):
        if self.processed is not None:
            filetypes = [
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
            path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)
            if path:
                try:
                    save_image(path, self.processed)
                    self.history.insert("end", f"Saved: {path.split('/')[-1]}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        else:
            messagebox.showwarning("No Image", "No image to save")

# ---- CREATE UPDATE IMAGE ON THE WINDOW FEATURE -----
    def update_image(self, img):
        if img is not None:
            self.update_idletasks()
            
            # Image Display Area
            frame_width = self.image_label.winfo_width()
            frame_height = self.image_label.winfo_height()
            
            if frame_width > 10 and frame_height > 10:
                # Get image 
                img_height, img_width = img.shape[:2]
                
                # Calculate scaling to fit
                width_ratio = frame_width / img_width
                height_ratio = frame_height / img_height
                scale = min(width_ratio, height_ratio)
                
                # Resize if needed
                if scale < 1 or scale > 1.5:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to Tkinter PhotoImage
            self.tk_img = cv_to_tk(img)
            self.image_label.config(image=self.tk_img)
            self.image_label.image = self.tk_img

# ---- COMBINE ALL FILTERS METHOD -----
    def apply_all_filters(self):
        if self.original is None:
            return
        
        # Start with base image, NOT the one with background removed
        if self.has_background_removed:
            temp = self.original.copy()
        else:
            temp = self.original.copy()
        
        # Apply global blur first
        if self.gaussian_value > 0:
            temp = gaussian_blur(temp, self.gaussian_value)
        
        if self.median_value > 0:
            temp = median_blur(temp, self.median_value)
        
        # Apply selective blur using optimized version
        temp = self.apply_selective_blur(temp)
        
        # Apply other filters
        if self.darken_value > 0:
            temp = adjust_darken(temp, self.darken_value)
            
        if self.brighten_value > 0:
            temp = adjust_brighten(temp, self.brighten_value)
        
        if self.is_grayscale:
            temp = grayscale(temp)
            
            if self.is_blackwhite:
                temp = black_white(temp, self.bw_threshold)
        
        # Background removal
        if self.has_background_removed:
            if self.background_method == "grabcut":
                temp = remove_background_grabcut(temp)
            elif self.background_method == "simple":
                temp = remove_background_simple(temp, self.bg_threshold)
            elif self.background_method == "edge":
                temp = remove_background_edge(temp)
        
        if self.show_binary:
            temp = show_binary_mask(temp)
        
        self.processed = temp
        self.update_image(self.processed)
    
# ---- CREATE A PREVIEW OF MASK BEING DRAWN ----
    def create_mask_preview(self, start, end):
        if self.processed is None or start is None or end is None:
            return None
        
        # Get image dimensions
        img_h, img_w = self.processed.shape[:2]
        
        # Get display dimensions
        label_h = self.image_label.winfo_height()
        label_w = self.image_label.winfo_width()
        
        # Calculate scaling factor
        scale_x = img_w / label_w if label_w > 0 else 1
        scale_y = img_h / label_h if label_h > 0 else 1
        
        # Convert screen coordinates to image coordinates
        x1 = int(start[0] * scale_x)
        y1 = int(start[1] * scale_y)
        x2 = int(end[0] * scale_x)
        y2 = int(end[1] * scale_y)
        
        # Create mask based on shape type: rectangle, circle and freeform
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if self.current_mask_type == "rectangle":
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        elif self.current_mask_type == "circle":
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5) // 2
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        elif self.current_mask_type == "freeform" and len(self.mask_points) > 2:
            # Convert all points
            img_points = []
            for px, py in self.mask_points:
                ix = int(px * scale_x)
                iy = int(py * scale_y)
                img_points.append([ix, iy])
            
            if len(img_points) > 2:
                pts = np.array(img_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        
        # Apply feathering for smooth edges
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        return mask

# ---- CREATE A BINARY MASK FROM DRAWING COORDINATES ----
    def create_mask(self, start, end):
        if self.processed is None or start is None or end is None:
            return None
        
        # Get image dimensions
        img_h, img_w = self.processed.shape[:2]
        
        # Get display dimensions
        label_h = self.image_label.winfo_height()
        label_w = self.image_label.winfo_width()
        
        # Calculate scaling factor
        scale_x = img_w / max(label_w, 1)  # Avoid division by zero
        scale_y = img_h / max(label_h, 1)
        
        # Convert screen coordinates to image coordinates
        x1 = int(start[0] * scale_x)
        y1 = int(start[1] * scale_y)
        x2 = int(end[0] * scale_x)
        y2 = int(end[1] * scale_y)
        
        # Create mask based on shape type
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if self.current_mask_type == "rectangle":
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        elif self.current_mask_type == "circle":
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5) // 2
            cv2.circle(mask, (center_x, center_y), max(radius, 1), 255, -1)
        
        elif self.current_mask_type == "freeform" and len(self.mask_points) > 2:
            # Convert all points
            img_points = []
            for px, py in self.mask_points:
                ix = int(px * scale_x)
                iy = int(py * scale_y)
                img_points.append([ix, iy])
            
            if len(img_points) > 2:
                pts = np.array(img_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        
        # Apply feathering for smooth edges
        if np.any(mask > 0):
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        return mask

# ---- SHOW MASK OVERLAY ON IMAGE ----
    def show_mask_preview(self, mask):
        if self.processed is None or mask is None:
            return None
        
        # Create a copy of the image for preview
        preview = self.processed.copy()
        
        # Create colored overlay (yellow with transparency)
        overlay = np.zeros_like(preview)
        overlay[mask > 0] = [0, 255, 255]  # Yellow color
        
        # Blend with original (30% opacity)
        alpha = 0.3
        preview = cv2.addWeighted(preview, 1 - alpha, overlay, alpha, 0)
        
        return preview

# ---- START DRAWING MASK FOR BLURRING  ----
    def start_mask_draw(self, event):
        if self.selective_blur_mode and self.processed is not None:
            self.mask_start = (event.x, event.y)
            self.mask_points = [self.mask_start]

# ---- UPDATE MASK DRAWING  ----
    def draw_mask(self, event):
        if self.selective_blur_mode and self.mask_start is not None:
            current_pos = (event.x, event.y)
            
            if self.current_mask_type == "freeform":
                self.mask_points.append(current_pos)
            
            # Create mask
            temp_mask = self.create_mask(self.mask_start, current_pos)
            if temp_mask is not None:
                # Show preview
                preview = self.show_mask_preview(temp_mask)
                if preview is not None:
                    temp_tk_img = cv_to_tk(preview)
                    self.image_label.config(image=temp_tk_img)
                    self.image_label.image = temp_tk_img

# ---- FINISH DRAWING MASK AND ADD TO HISTORY METHOD ----
    def finish_mask_draw(self, event):
        if self.selective_blur_mode and self.mask_start is not None:
            end_pos = (event.x, event.y)
            
            # Create final mask
            final_mask = self.create_mask(self.mask_start, end_pos)
            if final_mask is not None:
                self.current_mask = final_mask
                
                # Check if this is median blur
                if self.selective_blur_type == "median":
                    if self.median_blur_used:
                        messagebox.showwarning(
                            "Median Blur Restriction",
                            "Median blur can only be used once per session.\n"
                            "Please use Gaussian blur or reset the filters."
                        )
                        return  # Don't add this mask for Median Blur
                    else:
                        self.median_blur_used = True  # Set the flag
                
                # Add to history
                self.mask_history.append({
                    'mask': final_mask.copy(),
                    'blur_type': self.selective_blur_type,
                    'intensity': self.selective_intensity
                })
                
                # Apply filters to update display with actual blur
                self.apply_all_filters()
                self.history.insert("end", 
                    f"Added {self.current_mask_type} blur area ({self.selective_blur_type}, intensity: {self.selective_intensity})")
            
            # Reset for next drawing
            self.mask_start = None
            self.mask_points = []

# ---- APPLYING SELECTIVE BLUR USING ROI EXTRACTION ----
    def apply_selective_blur(self, image):
    
        if not self.mask_history:
            return image.copy()
        
        result = image.copy()
        
        # Separate masks by blur type 
        gaussian_masks = []
        median_masks = []
        
        for mask_data in self.mask_history:
            if mask_data['blur_type'] == 'gaussian':
                gaussian_masks.append(mask_data)
            else:
                median_masks.append(mask_data)
        
        # Process Gaussian blurs
        if gaussian_masks:
            result = self._apply_gaussian_masks_roi(result, gaussian_masks)
        
        # Process Median blurs 
        if median_masks:
            result = self._apply_median_masks_roi(result, median_masks)
        
        return result

# ---- APPLY GAUSSIAN BLUR FROM MASKING (TO ROI AREA ONLY) ----
    def _apply_gaussian_masks_roi(self, image, masks):
        result = image.copy()
        
        for mask_data in masks:
            mask = mask_data['mask']
            intensity = mask_data['intensity']
            
            # Calculate kernel size
            kernel_size = max(1, (intensity * 51) // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Skip if mask is empty
            if not np.any(mask > 0):
                continue
            
            # Get ROI bounding box 
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            y1, y2 = y_indices[0], y_indices[-1]
            x1, x2 = x_indices[0], x_indices[-1]
            
            # Extract only the ROI
            roi_height = y2 - y1 + 1
            roi_width = x2 - x1 + 1
            
            # Skip very small ROIs (no blur effect)
            if roi_height < 3 or roi_width < 3:
                continue
            
            roi = result[y1:y2+1, x1:x2+1].copy()
            mask_roi = mask[y1:y2+1, x1:x2+1]
            
            # Blur only the ROI
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
            # Apply mask within ROI only
            mask_normalized = mask_roi.astype(np.float32) / 255.0
            
            if len(roi.shape) == 3:  # Color image
                mask_3d = np.stack([mask_normalized] * 3, axis=2)
                blended_roi = roi * (1 - mask_3d) + blurred_roi * mask_3d
            else:  # Grayscale
                blended_roi = roi * (1 - mask_normalized) + blurred_roi * mask_normalized
            
            # Paste back only the blended ROI
            result[y1:y2+1, x1:x2+1] = blended_roi
        
        return result
    
# ---- APPLY MEDIAAN BLUR FROM MASKING ----
    def _apply_median_masks_roi(self, image, masks):

        result = image.copy()
        
        # Group by kernel size to potentially cache results
        masks_by_kernel = {}
        for mask_data in masks:
            intensity = mask_data['intensity']
            kernel_size = max(1, (intensity * 51) // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            if kernel_size not in masks_by_kernel:
                masks_by_kernel[kernel_size] = []
            masks_by_kernel[kernel_size].append(mask_data)
        
        # Process each kernel size separately
        for kernel_size, mask_list in masks_by_kernel.items():
            # Warn for large median kernels
            if kernel_size > 15 and len(mask_list) > 1:
            # Suppposed to show a warning message here
                pass
            
            for mask_data in mask_list:
                mask = mask_data['mask']
                
                # Skip if mask is empty
                if not np.any(mask > 0):
                    continue
                
                # Get ROI bounding box
                rows = np.any(mask > 0, axis=1)
                cols = np.any(mask > 0, axis=0)
                
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                y1, y2 = y_indices[0], y_indices[-1]
                x1, x2 = x_indices[0], x_indices[-1]
                
                # Calculate ROI size
                roi_height = y2 - y1 + 1
                roi_width = x2 - x1 + 1
                
                # Skip very small ROIs
                if roi_height < kernel_size or roi_width < kernel_size:
                    continue
                
                # Extract ROI
                roi = result[y1:y2+1, x1:x2+1].copy()
                mask_roi = mask[y1:y2+1, x1:x2+1]
                
                # Apply median blur to ROI
                blurred_roi = cv2.medianBlur(roi, kernel_size)
                
                # Apply mask within ROI
                mask_normalized = mask_roi.astype(np.float32) / 255.0
                
                if len(roi.shape) == 3:
                    mask_3d = np.stack([mask_normalized] * 3, axis=2)
                    blended_roi = roi * (1 - mask_3d) + blurred_roi * mask_3d
                else:
                    blended_roi = roi * (1 - mask_normalized) + blurred_roi * mask_normalized
                
                # Paste back
                result[y1:y2+1, x1:x2+1] = blended_roi
        
        return result

# ---- CLEAR SELECTIVE BLUR AREA ----
    def clear_selective_areas(self):
        self.mask_history = []
        self.current_mask = None
        self.apply_all_filters()
        self.history.insert("end", "Cleared all selective blur areas")

# ---- REMOVE THE LATEST SELECTIVE BLURRED AREA ----
    def undo_last_mask(self):
        """Remove the last selective blur area."""
        if self.mask_history:
            self.mask_history.pop()
            self.apply_all_filters()
            self.history.insert("end", "Removed last selective blur area")

# ---- STYLING FOR SELECTIVE MODE AND WARNING MESSAGE FOR SELECTIVE MEDIAN BLURRING ----
# --- TOGGLE FOR SELECTIVE MODE ON/OFF
    def toggle_selective_mode(self):
        """Toggle selective blur mode on/off."""
        self.selective_blur_mode = not self.selective_blur_mode
        
        if self.selective_blur_mode:
            # Check if median blur was already used
            if self.median_blur_used:
                messagebox.showwarning(
                    "Median Blur Restriction",
                    "Median blur can only be used once per session.\n"
                    "Please use Gaussian blur for additional areas or reset the filters."
                )
                self.selective_blur_mode = False  # Keep it disabled
                return  # Don't enable selective mode
            
            self.selective_toggle_btn.config(text="SELECTIVE MODE ACTIVE", 
                                            bg="#4a4a4a", fg="white")
            self.history.insert("end", "Selective blur mode enabled")
            
            # Use processed image instead of original
            if self.processed is not None:
                self.update_image(self.processed)
        else:
            self.selective_toggle_btn.config(text="ENABLE SELECTIVE MODE", 
                                            bg="#2a2a2a", fg="white")
            self.history.insert("end", "Selective blur mode disabled")
            
            # Clear mask drawing state
            self.mask_start = None
            self.mask_points = []
            
            # Show current image (should have blur applied)
            if self.processed is not None:
                self.update_image(self.processed)

# ---- TOGGLE CROP MODE ON/OFF ----
    def toggle_crop_mode(self):
        self.crop_mode = not self.crop_mode
        
        if self.crop_mode:
            self.crop_toggle_btn.config(text="CROP MODE ACTIVE", 
                                    bg="#4a4a4a", fg="white")
            self.history.insert("end", "Crop mode enabled - Ctrl+Click and drag on image")
            
            # Disable selective blur mode if active
            if self.selective_blur_mode:
                self.selective_blur_mode = False
                self.selective_toggle_btn.config(text="ENABLE SELECTIVE MODE", 
                                                bg="#2a2a2a", fg="white")
            
            # Show current image
            if self.processed is not None:
                self.update_image(self.processed)
        else:
            self.crop_toggle_btn.config(text="ENABLE CROP", 
                                    bg="#2a2a2a", fg="white")
            self.history.insert("end", "Crop mode disabled")
            
            # Clear crop selection
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None
            
            # Show current image without crop overlay
            if self.processed is not None:
                self.update_image(self.processed)

# ---- CROP WITH RECTANGLE (INITIALIZED) ----
    def start_crop(self, event):
        """Start drawing crop rectangle."""
        if self.crop_mode and self.processed is not None:
            self.crop_start = (event.x, event.y)
            self.crop_end = (event.x, event.y)

# ---- UPDATE CROPPING AREA AS USER DRAGS THE MOUSE ----
    def draw_crop(self, event):
        if self.crop_mode and self.crop_start is not None:
            self.crop_end = (event.x, event.y)
            
            # Show preview
            self.show_crop_preview()

# ---- APPLY CROPPING ----
    def finish_crop(self, event):
        """Finish crop selection and apply."""
        if self.crop_mode and self.crop_start is not None:
            self.crop_end = (event.x, event.y)
            
            # Apply crop
            self.apply_crop()
            
            # Reset crop state
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None

# ---- DISPLAY THE CROPPED VISUAL RECTANGLE OVERLAY ON IMAGE ----
    def show_crop_preview(self):
        if self.processed is None or self.crop_start is None or self.crop_end is None:
            return
        
        # Create a copy of the image for preview
        preview = self.processed.copy()
        
        # Get image dimensions
        img_h, img_w = self.processed.shape[:2]
        
        # Get display dimensions
        label_h = self.image_label.winfo_height()
        label_w = self.image_label.winfo_width()
        
        # Calculate scaling factor
        scale_x = img_w / max(label_w, 1)
        scale_y = img_h / max(label_h, 1)
        
        # Convert screen coordinates to image coordinates
        x1 = int(self.crop_start[0] * scale_x)
        y1 = int(self.crop_start[1] * scale_y)
        x2 = int(self.crop_end[0] * scale_x)
        y2 = int(self.crop_end[1] * scale_y)
        
        # Store crop rectangle
        self.crop_rect = {
            'x1': min(x1, x2),
            'y1': min(y1, y2),
            'x2': max(x1, x2),
            'y2': max(y1, y2)
        }
        
        # Draw crop rectangle on preview
        cv2.rectangle(preview, 
                    (self.crop_rect['x1'], self.crop_rect['y1']),
                    (self.crop_rect['x2'], self.crop_rect['y2']),
                    (0, 255, 0),  # Green color
                    2)  # Thickness
        
        # Draw crosshair at center
        center_x = (self.crop_rect['x1'] + self.crop_rect['x2']) // 2
        center_y = (self.crop_rect['y1'] + self.crop_rect['y2']) // 2
        cv2.drawMarker(preview, (center_x, center_y), (0, 255, 0), 
                    cv2.MARKER_CROSS, 20, 2)
        
        # Display dimensions
        width = self.crop_rect['x2'] - self.crop_rect['x1']
        height = self.crop_rect['y2'] - self.crop_rect['y1']
        
        # Add dimension text
        text = f"{width} x {height}"
        cv2.putText(preview, text, 
                (self.crop_rect['x1'] + 10, self.crop_rect['y1'] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update display
        temp_tk_img = cv_to_tk(preview)
        self.image_label.config(image=temp_tk_img)
        self.image_label.image = temp_tk_img

# APPLY CROPPING ON IMAGE
    def apply_crop(self):
        if self.processed is None or self.crop_rect is None:
            return
        
        try:
            # Get crop coordinates
            x1 = max(0, self.crop_rect['x1'])
            y1 = max(0, self.crop_rect['y1'])
            x2 = min(self.processed.shape[1], self.crop_rect['x2'])
            y2 = min(self.processed.shape[0], self.crop_rect['y2'])
            
            # Ensure valid rectangle
            if x2 <= x1 or y2 <= y1:
                messagebox.showerror("Invalid Crop", 
                                    "Crop rectangle is too small or invalid")
                return
            
            # Perform crop on PROCESSED image only
            cropped = self.processed[y1:y2, x1:x2]
            
            # Update processed image ONLY
            self.processed = cropped
             
            # Update display
            self.update_image(self.processed)
            
            # Update history
            width = x2 - x1
            height = y2 - y1
            self.history.insert("end", f"Cropped to: {width}x{height} pixels")
            
            # Reset crop mode
            self.crop_mode = False
            if hasattr(self, 'crop_toggle_btn'):
                self.crop_toggle_btn.config(text="ENABLE CROP", 
                                            bg="#2a2a2a", fg="white")
            
        except Exception as e:
            messagebox.showerror("Crop Error", f"Failed to crop image: {str(e)}")

# ---- ALLOW CROPPING TO THE ASPECT RATIO AVAILABLE ----
    def crop_to_aspect_ratio(self, aspect_ratio):
        if self.processed is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        try:
            h, w = self.processed.shape[:2]
            current_ratio = w / h
            
            if current_ratio > aspect_ratio:
                # Image is wider than target ratio, crop width
                new_width = int(h * aspect_ratio)
                x1 = (w - new_width) // 2
                x2 = x1 + new_width
                y1, y2 = 0, h
            else:
                # Image is taller than target ratio, crop height
                new_height = int(w / aspect_ratio)
                y1 = (h - new_height) // 2
                y2 = y1 + new_height
                x1, x2 = 0, w
            
            # Perform crop on PROCESSED image only
            cropped = self.processed[y1:y2, x1:x2]
            
            # Update processed image ONLY
            self.processed = cropped
            
            # DO NOT update original
            
            # Update display
            self.update_image(self.processed)
            
            # Update history
            new_width = x2 - x1
            new_height = y2 - y1
            ratio_name = {
                1: "1:1 (Square)",
                4/3: "4:3 (Standard)",
                16/9: "16:9 (Wide)",
                3/2: "3:2 (35mm Film)"
            }.get(aspect_ratio, f"{aspect_ratio:.2f}:1")
            self.history.insert("end", f"Aspect ratio crop: {ratio_name} ({new_width}x{new_height})")
            
        except Exception as e:
            messagebox.showerror("Crop Error", f"Failed to crop image: {str(e)}")  
            
# ---- DEFINE METHOD TO APPLY GAUSSIAN BLUR -----
    def apply_gaussian(self, v):
        self.gaussian_value = int(v)
        self.apply_all_filters()
        if int(v) > 0:
            self.history.insert("end", f"Gaussian Blur: {v}")

# ---- DEFINE METHOD TO APPLY MEDIAN BLUR -----
    def apply_median(self, v):
        self.median_value = int(v)
        self.apply_all_filters()
        if int(v) > 0:
            self.history.insert("end", f"Median Blur: {v}")

# ---- DEFINE METHOD TO APPLY GRAYSCALE FILTER -----
    def apply_gray(self):
        self.is_grayscale = not self.is_grayscale
        
        # Update button appearance
        if self.is_grayscale:
            self.grayscale_btn.config(bg="#4a4a4a", fg="white")
        else:
            self.grayscale_btn.config(bg="#2a2a2a", fg="white")
            # Turn off B&W if turning off grayscale
            if self.is_blackwhite:
                self.is_blackwhite = False
                self.bw_toggle_btn.config(text="BLACK & WHITE", bg="#2a2a2a", fg="white")
                self.bw_slider.config(state="disabled")
        
        self.apply_all_filters()
        status = "ON" if self.is_grayscale else "OFF"
        self.history.insert("end", f"Grayscale: {status}")

# ---- DEFINE METHOD TO ENABLE BUTTON FOR BLACK AND WHITE FILTER -----
    def toggle_blackwhite(self):
        if not self.is_grayscale:
            self.history.insert("end", "Enable Grayscale first for B&W")
            return
            
        self.is_blackwhite = not self.is_blackwhite
        
        # button appearance
        if self.is_blackwhite:
            self.bw_toggle_btn.config(text="BLACK & WHITE ✓", bg="#4a4a4a", fg="white")
            self.bw_slider.config(state="normal")
        else:
            self.bw_toggle_btn.config(text="BLACK & WHITE", bg="#2a2a2a", fg="white")
            self.bw_slider.config(state="disabled")
            
        self.apply_all_filters()
        status = "ON" if self.is_blackwhite else "OFF"
        self.history.insert("end", f"Black & White: {status}")

# ---- DEFINE METHOD TO APPLY BLACK AND WHITE FILTER -----
    def apply_bw(self, v):
        self.bw_threshold = int(v)
        self.apply_all_filters()
        self.history.insert("end", f"B&W Threshold: {v}")

# ---- DEFINE METHOD TO APPLY DARKENING FILTER -----
    def apply_darken(self, v):
        self.darken_value = int(v)
        self.apply_all_filters()
        if int(v) > 0:
            self.history.insert("end", f"Darken: {v}")

# ---- DEFINE METHOD TO APPLY BRIGHTENING FILTER -----
    def apply_brighten(self, v):
        self.brighten_value = int(v)
        self.apply_all_filters()
        if int(v) > 0:
            self.history.insert("end", f"Brighten: {v}")

# ---- DEFINE METHOD TO APPLY SEGMENTATION - BACKGROUND REMOVAL -----
    def apply_background_removal(self, method):
        if self.original is not None:
            # Start from original for background removal
            if method == "grabcut":
                self.processed = remove_background_grabcut(self.original)
                self.history.insert("end", "Background removed (GrabCut)")
            elif method == "simple":
                self.processed = remove_background_simple(self.original, self.bg_threshold)
                self.history.insert("end", f"Background removed (Simple, threshold: {self.bg_threshold})")
            elif method == "edge":
                self.processed = remove_background_edge(self.original)
                self.history.insert("end", "Background removed (Edge-based)")
            
            # Track background removal state
            self.has_background_removed = True
            self.background_method = method
            
            # Reset filter states since background removal replaces the image
            self.reset_filter_states()
            
            self.update_image(self.processed)
            
# ---- DEFINE METHOD TO APPLY RESIZE FILTER -----
    def apply_resize(self):
        if self.processed is not None:  # Use processed image instead of original
            width_str = self.width_var.get()
            height_str = self.height_var.get()
            
            try:
                width = int(width_str) if width_str else None
                height = int(height_str) if height_str else None
                
                if width or height:
                    # Resize the current processed image
                    self.processed = resize_image(self.processed, width, height)
                    
                    # Update display with resized image
                    self.update_image(self.processed)
                    
                    if width and height:
                        self.history.insert("end", f"Resized to: {width}x{height}")
                    elif width:
                        self.history.insert("end", f"Resized width to: {width}")
                    elif height:
                        self.history.insert("end", f"Resized height to: {height}")
            except ValueError:
                self.history.insert("end", "Invalid resize values")
                messagebox.showerror("Invalid Input", "Please enter valid numbers for width and height")
        else:
            messagebox.showwarning("No Image", "Please load an image first")

# ---- DEFINE METHOD TO APPLY PRESET SIZES -----
    def apply_preset_size(self, preset):
        if self.processed is not None:  # Use processed image instead of original
            # Resize the current processed image
            self.processed = resize_to_preset(self.processed, preset)
            
            # Update display with resized image
            self.update_image(self.processed)
            
            preset_names = {
                "instagram": "Instagram (1080x1080)",
                "facebook": "Facebook (1200x630)",
                "twitter": "Twitter (1200x675)",
                "hd": "HD (1280x720)",
                "full_hd": "Full HD (1920x1080)"
            }
            
            if preset in preset_names:
                self.history.insert("end", f"Resized to: {preset_names[preset]}")
        else:
            messagebox.showwarning("No Image", "Please load an image first")

# ---- DEFINE METHOD TO UPDATE THE BACKGROUND THRESHOLD -----
    def update_bg_threshold(self, v):
        self.bg_threshold = int(v)

# ---- DEFINE METHOD TO ENABLE BUTTON FOR BINARY MASK -----
    def toggle_binary_mask(self):
        self.show_binary = not self.show_binary
        if self.show_binary:
            self.binary_toggle_btn.config(text="HIDE BINARY MASK", bg="#4a4a4a", fg="white")
        else:
            self.binary_toggle_btn.config(text="SHOW BINARY MASK", bg="#2a2a2a", fg="white")
        
        self.apply_all_filters()
        status = "ON" if self.show_binary else "OFF"
        self.history.insert("end", f"Binary Mask: {status}")

# ---- DEFINE METHOD TO CONFIGURE RESIZE TAB WITH CROPPING AND RESIZING FEATURES
    def resize_tab(self, notebook):
        tab = tk.Frame(notebook, bg="#1e1e1e")
        notebook.add(tab, text="RESIZE")
        
        tk.Label(tab, text="Image Resizing", font=("Arial", 12, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        # Crop frames
        crop_frame = tk.LabelFrame(tab, text="Crop Tools", 
                                  font=("Arial", 10),
                                  fg="white", bg="#1e1e1e",
                                  relief="ridge")
        crop_frame.pack(fill="x", padx=10, pady=5)
        
        # Crop toggle button
        self.crop_toggle_btn = tk.Button(crop_frame, 
                                        text="ENABLE CROP",
                                        command=self.toggle_crop_mode,
                                        font=("Arial", 9),
                                        width=20, height=1,
                                        bg="#2a2a2a", fg="white")
        self.crop_toggle_btn.pack(pady=10)
        
        tk.Label(crop_frame, 
                text="Ctrl+Click and drag on image to select crop area",
                fg="#888888", bg="#1e1e1e", font=("Arial", 8)).pack(pady=2)
        
        # Quick crop buttons
        quick_crop_frame = tk.Frame(crop_frame, bg="#1e1e1e")
        quick_crop_frame.pack(pady=10, fill="x", padx=20)
        
        # Create buttons with grid layout
        button1 = tk.Button(quick_crop_frame, text="Square (1:1)", 
                            command=lambda: self.crop_to_aspect_ratio(1),
                            width=15, height=1,
                            bg="#2a2a2a", fg="white")
        button1.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        
        button2 = tk.Button(quick_crop_frame, text="Wide (16:9)", 
                            command=lambda: self.crop_to_aspect_ratio(16/9),
                            width=15, height=1,
                            bg="#2a2a2a", fg="white")
        button2.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        # Reset crop button
        reset_button = tk.Button(quick_crop_frame, text="RESET CROP", 
                                command=self.reset_crop_only,
                                width=15, height=1,
                                bg="#2a2a2a", fg="white")
        reset_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")
        
        # Configure grid columns to expand equally
        quick_crop_frame.grid_columnconfigure(0, weight=1)
        quick_crop_frame.grid_columnconfigure(1, weight=1)
        quick_crop_frame.grid_columnconfigure(2, weight=1)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Existing resize section
        tk.Label(tab, text="Resize Image", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=5)
        
        # Width input
        tk.Label(tab, text="Width (pixels):", fg="white", bg="#1e1e1e").pack(pady=5)
        self.width_var = tk.StringVar()
        width_entry = tk.Entry(tab, textvariable=self.width_var, 
                                width=20, bg="#2a2a2a", fg="white",
                                insertbackground="white")
        width_entry.pack(pady=5)
        
        # Height input
        tk.Label(tab, text="Height (pixels):", fg="white", bg="#1e1e1e").pack(pady=10)
        self.height_var = tk.StringVar()
        height_entry = tk.Entry(tab, textvariable=self.height_var, 
                                width=20, bg="#2a2a2a", fg="white",
                                insertbackground="white")
        height_entry.pack(pady=5)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Apply button
        tk.Button(tab, text="APPLY RESIZE", 
                    command=self.apply_resize,
                    font=("Arial", 10, "bold"),
                    width=15, height=2,
                    bg="#2a2a2a", fg="white").pack(pady=10)
        
        tk.Frame(tab, height=2, bg="#333333").pack(fill="x", padx=20, pady=15)
        
        # Preset sizes
        tk.Label(tab, text="Preset Sizes:", font=("Arial", 10, "bold"),
                fg="white", bg="#1e1e1e").pack(pady=10)
        
        presets_frame = tk.Frame(tab, bg="#1e1e1e")
        presets_frame.pack(pady=5)
        
        presets = [
            ("Instagram (1080x1080)", "instagram"),
            ("Facebook (1200x630)", "facebook"),
            ("Twitter (1200x675)", "twitter"),
            ("HD (1280x720)", "hd"),
            ("Full HD (1920x1080)", "full_hd")
        ]
        
        for name, preset in presets:
            btn = tk.Button(presets_frame, text=name,
                            command=lambda p=preset: self.apply_preset_size(p),
                            font=("Arial", 8),
                            width=20, height=1,
                            bg="#2a2a2a", fg="white")
            btn.pack(pady=2)

# ---- DEFINE METHOD TO APPLY GRAYSCALE FILTER -----
    def apply_preset_size(self, preset):
        if self.processed is not None:  # Use processed image instead of original
            # Resize the current processed image
            self.processed = resize_to_preset(self.processed, preset)
            
            # Update display with resized image
            self.update_image(self.processed)
            
            preset_names = {
                "instagram": "Instagram (1080x1080)",
                "facebook": "Facebook (1200x630)",
                "twitter": "Twitter (1200x675)",
                "hd": "HD (1280x720)",
                "full_hd": "Full HD (1920x1080)"
            }
            
            if preset in preset_names:
                self.history.insert("end", f"Resized to: {preset_names[preset]}")
        else:
            messagebox.showwarning("No Image", "Please load an image first")

# ---- DEFINE METHOD TO RESET FILTERS AND REVERT TO ORIGINAL IMAGE -----
    def reset_filters(self):
        # Clear history
        self.history.delete(0, tk.END)
        
        # Reset all filter values
        self.gaussian_value = 0
        self.median_value = 0
        self.darken_value = 0
        self.brighten_value = 0
        self.is_grayscale = False
        self.is_blackwhite = False
        self.bw_threshold = 127
        self.bg_threshold = 240
        self.show_binary = False
        
        # Reset background removal state
        self.has_background_removed = False
        self.background_method = None
        
        # Reset UI controls
        if hasattr(self, 'gaussian_slider'):
            self.gaussian_slider.set(0)
            
        if hasattr(self, 'median_slider'):
            self.median_slider.set(0)
            
        if hasattr(self, 'darken_slider'):
            self.darken_slider.set(0)
            
        if hasattr(self, 'brighten_slider'):
            self.brighten_slider.set(0)
            
        if hasattr(self, 'bw_slider'):
            self.bw_slider.set(127)
            self.bw_slider.config(state="disabled")
            
        if hasattr(self, 'bg_threshold_slider'):
            self.bg_threshold_slider.set(240)
            
        if hasattr(self, 'width_var'):
            self.width_var.set("")
            
        if hasattr(self, 'height_var'):
            self.height_var.set("")
            
        if hasattr(self, 'grayscale_btn'):
            self.grayscale_btn.config(bg="#2a2a2a", fg="white")
            
        if hasattr(self, 'bw_toggle_btn'):
            self.bw_toggle_btn.config(text="BLACK & WHITE", bg="#2a2a2a", fg="white")
            
        if hasattr(self, 'binary_toggle_btn'):
            self.binary_toggle_btn.config(text="SHOW BINARY MASK", bg="#2a2a2a", fg="white")
            
        # Reset image
        if self.original is not None:
            self.processed = self.original.copy()
            self.update_image(self.processed)
            self.history.insert("end", "All filters reset")
            self.placeholder_label.pack_forget()
        else:
            self.history.insert("end", "Filters reset")
            self.placeholder_label.pack(fill="both", expand=True)
            
        self.selective_blur_mode = False
        if hasattr(self, 'selective_toggle_btn'):
            self.selective_toggle_btn.config(text="ENABLE SELECTIVE MODE", 
                                            bg="#2a2a2a", fg="white")
        self.current_mask = None
        self.mask_start = None
        self.mask_points = []
        self.mask_history = []
        self.selective_intensity = 50
        if hasattr(self, 'selective_intensity_slider'):
            self.selective_intensity_slider.set(50)
        
        # Reset the median blur message flag
        self.median_blur_used = False
        
        # Reset crop
        self.crop_mode = False
        if hasattr(self, 'crop_toggle_btn'):
            self.crop_toggle_btn.config(text="ENABLE CROP", 
                                    bg="#2a2a2a", fg="white")
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        
# ---- COMBINE ALL METHOD IN 'app' -----
if __name__ == "__main__":
    app = SnappicApp()
    app.mainloop()