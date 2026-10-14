# 🏏 Indian Cricket Team Member Face ID

A beginner-friendly computer-vision project built with **Python, OpenCV, TensorFlow/MobileNetV2, and Streamlit**.

## Features
- Stylish dark blue/teal dashboard
- Upload JPG/JPEG/PNG images
- OpenCV face detection and face crop
- Train a multi-class classifier using your own player-image folders
- Prediction confidence and per-player scores

> Educational demo only. The prediction is not verified identity recognition. Results depend heavily on dataset quality and may be wrong.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- VS Code
- Internet for the first TensorFlow setup and MobileNetV2 pretrained weights

## 2. Open in VS Code
1. Extract this ZIP.
2. In VS Code, choose **File → Open Folder** and select `Indian_Cricket_Face_ID`.
3. Open **Terminal → New Terminal**.

## 3. Create a virtual environment (Windows)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If `py -3.11` isn't available, install Python 3.10 or 3.11 from python.org and select it in VS Code. If PowerShell blocks activation, use Command Prompt terminal and run:
```bat
.venv\Scripts\activate.bat
```

## 4. Add training images
Put images into separate folders inside `dataset`, e.g.
```text
dataset/
  Virat_Kohli/
    kohli1.jpg
    kohli2.jpg
  Rohit_Sharma/
    rohit1.jpg
    rohit2.jpg
```
Use at least **two players**, and preferably **20–50+ images per player**. Use a mix of clear poses and lighting. Do not put all players in one folder.

Dataset sources to explore: Kaggle (search “Indian cricketers images dataset” / individual player datasets) and Google Dataset Search. Check each dataset's licence and download rules. This project does not download a dataset automatically.

## 5. Train the model
```powershell
python train_model.py
```
Training can take a while and may download MobileNetV2's ImageNet weights the first time. When successful, it creates:
- `models/cricket_face_model.keras`
- `models/labels.json`

## 6. Launch the app
```powershell
streamlit run app.py
```
Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser. Upload a new image to test.

## Troubleshooting
- **No model found:** run `python train_model.py` first and check that `dataset` has at least two player folders with images.
- **No module named ...:** activate `.venv` and run `pip install -r requirements.txt`.
- **TensorFlow install error:** use 64-bit Python 3.10/3.11, then recreate the virtual environment.
- **Poor predictions:** add more varied images, balance the number of images per player, and use a clear single-person photo.
- **Face not detected:** the app falls back to classifying the full image, but results may be weaker.

## How it works
1. OpenCV Haar cascade tries to locate a face in the uploaded image.
2. MobileNetV2 pretrained on ImageNet extracts visual features.
3. A small trainable classification layer learns to predict the player folders.
4. Streamlit displays the top prediction and scores.

## Suggested project presentation
**Problem:** identify which of the selected trained cricket-player classes an uploaded image most resembles.  
**Input:** image file.  
**Output:** predicted class label and confidence score.  
**Tech stack:** Python, OpenCV, TensorFlow/Keras, MobileNetV2, Streamlit.  
**Future enhancements:** face embeddings, more robust face alignment, larger curated datasets, and evaluation using a held-out test set.
