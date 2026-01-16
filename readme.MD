# SNAPPIC - Enhance Pic to Its Peak

<div align="center">

**Slay your image edits like a pro!** 

</div>

## 🚀 What's the Vibe? 

### ✨ Key Features That Hit Different:

- **🎨 Smart Background Removal** - Yeet unwanted backgrounds like it's nothing!
- **🌀 Selective Blurring** - Make your subject pop while blurring the rest (dramatic effect unlocked! 📸)
- **🌈 Color Magic** - Grayscale? Black & White? We're serving looks!
- **✨ Light Adjustments** - Darken, brighten, make it glow like you're in a K-drama!
- **✂️ Crop & Resize** - For that perfect aspect ratio (IG square? Twitter wide? We got it!)
- **🔍 Binary Masks** - Create cool black-and-white effects for your aesthetic!

## 🎮 Quick Start - Let's Get Lit!

### Prerequisites (What You Need):

```bash
# The basics
pip install opencv-python pillow numpy
```

### Installation (No Cap, It's Easy):

1. **Clone this vibe:**
```bash
git clone https://github.com/yourusername/snappic.git
cd snappic
```

2. **Run the main event:**
```bash
python main.py
```

3. **Watch the magic happen!** ✨

## 🖼️ How to Use:

### 1. **Load Your Pic** 📤
   - Click "OPEN" or drag & drop
   - Supports: JPG, PNG, BMP, TIFF (we're inclusive like that!)

### 2. **Edit Like a Pro** 🎨

#### **Blur Tab** 🌫️
- **Global Blur**: Smooth out everything
- **Selective Blur**: Highlight what matters, blur the rest (big main character energy!)

#### **Color Tab** 🎨
- **Grayscale**: Classic black & white aesthetic
- **Black & White**: Adjust thresholds for that perfect contrast

#### **Light Tab** 💡
- **Brighten**: Make it pop!
- **Darken**: Add some mystery

#### **Segmentation Tab** ✂️
- **Background Removal**: Three methods to delete backgrounds (no green screen needed!)
- **Binary Masks**: Create cool threshold effects

#### **Resize Tab** 📏
- **Crop Tools**: Square, widescreen, or custom crop
- **Preset Sizes**: IG, Twitter, Facebook ready!
- **Reset Crop**: Oopsie? No problem!

### 3. **Save Your Masterpiece** 💾
- Save as PNG (with transparency!), JPG, or BMP
- **Pro Tip**: Save binary masks separately for extra creative projects!

## 🎯 Pro Tips (The Secret Sauce):

1. **Selective Blur Hack**: 
   - Enable selective mode → Draw shapes → Watch the magic!
   - Use different shapes (rectangle, circle, freeform) for unique effects

2. **Background Removal Pro Moves**:
   - **GrabCut**: Best for complex backgrounds
   - **Simple**: Perfect for white/light backgrounds
   - **Edge-Based**: For those crisp, clean cuts

3. **Binary Mask Flex**:
   - Adjust the threshold slider for different effects
   - Save masks separately for layered edits

4. **Keyboard Shortcuts**:
   - `Ctrl + Click & Drag` = Crop selection
   - Regular click & drag = Selective blur drawing

## 🏗️ Project Structure (For the Nerds 🤓):

```
SnapPic/
├── main.py              # Main entry point
├── app.py               # Main application (GUI + logic)
├── processing/          # Image processing magic
│   ├── blur.py          # Blur functions
│   ├── light.py         # Brightness adjustments
│   ├── tone.py          # Color operations
│   └── segmentation.py  # Background removal & masks
└── utils/
    └── image_io.py      # Image loading/saving
```

## 🛠️ Tech Stack (The Building Blocks):

- **Python 3.8+** - Because we're modern like that
- **OpenCV** - The image processing powerhouse
- **Tkinter** - Clean, dark-mode GUI that's easy on the eyes
- **NumPy** - Math wizardry for pixel perfection
- **Pillow** - Image conversion magic

## 🎨 UI/UX Features:

- **Dark Mode Default** - Because light mode is so 2010
- **Real-time Previews** - See changes as you make them
- **History Log** - Track your edits (in case you want to recreate that perfect look)
- **Intuitive Sliders** - Smooth controls for precise adjustments
- **Clean Layout** - Everything where you expect it to be

## 🤝 Contributing (Want to Add Your Flavor?)

We love collabs! Here's how you can contribute:

1. **Fork it** 🍴
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request** 🎉

### Contribution Ideas:
- Add new filters (vintage, sepia, etc.)
- Implement batch processing
- Add social media templates
- Create preset filter packs

## 🙌 Shoutouts

Made with ❤️ by the SNAPPIC squad. Special thanks to:
- OpenCV community for the amazing library
- All the beta testers who helped shape this tool
- You, for using our editor! 😊

## 🆘 Need Help?

- **Bug Reports**: Open an issue on GitHub
- **Feature Requests**: We'd love to hear your ideas!
- **Just Want to Chat**: Reach out and say hi!

---

<div align="center">

**Remember: Your creativity has no limits! Go forth and create something amazing!** 🎨✨
**Made with pixels, passion, and a whole lot of Python magic!** 🐍✨

</div>

## 📸 Screenshots (Coming Soon!)

---

**P.S.** If you make something cool with SNAPPIC, do PM us! We'd love to see your creations! 📱✨
