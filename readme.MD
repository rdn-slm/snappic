## snappic/
│
├── main.py                    # Entry point
│
├── ui/
│   ├── app.py                 # Main Tkinter window
│   ├── panels.py              # Left (history), center (image), right (controls)
│   └── styles.py              # Colors, fonts, constants
│
├── processing/
│   ├── blur.py                # Gaussian & Median blur
│   ├── tone.py                # Grayscale & B/W
│   └── light.py               # Darken & brighten
│
├── utils/
│   ├── image_io.py            # Open, save, convert
│   └── history.py             # Edit history tracking
│
├── assets/
│   └── icons/                # Optional (buttons/icons)
│
├── output/                    # Saved images
│
└── README.md
