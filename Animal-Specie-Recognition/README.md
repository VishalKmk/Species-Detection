# Animal Species Recognition

A simple Streamlit app for image-based animal classification using pretrained TensorFlow/Keras models.

The current app uses `ResNet50` with ImageNet weights to predict the top class for an uploaded image, then stores the prediction history in `results.csv`. The repository also includes a separate `VGG16` experiment module.

## Overview

This project was built as a lightweight final-year project prototype to explore computer vision with:

- Streamlit for the user interface
- TensorFlow / Keras for pretrained image classification
- Pandas for viewing saved prediction history
- CSV logging for simple result tracking

It is best described as an **animal image classification demo powered by pretrained ImageNet models**, not a custom-trained species recognition pipeline.

## Features

- Upload an image in `jpg`, `jpeg`, or `png` format
- Predict the top class label using `ResNet50`
- Display prediction confidence
- Save predictions with date and time to `results.csv`
- View previous saved results inside the sidebar
- Includes a separate `VGG16` experiment in `vg16.py`

## Tech Stack

- Python `3.12`
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Pillow

## Project Structure

```text
Animal-Specie-Recognition/
|-- app.py              # Main Streamlit app
|-- vg16.py             # VGG16-based experiment module
|-- requirements.txt    # Python dependencies
|-- results.csv         # Saved prediction history
|-- test.ipynb
|-- testmodel.ipynb
|-- venv/
```

## Setup

### 1. Clone or open the project

Open the project folder in your terminal or VS Code.

### 2. Create a virtual environment

```powershell
py -3.12 -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the App

```powershell
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. Upload an image.
2. The app resizes it to `224 x 224`.
3. `ResNet50` preprocesses the image and generates predictions.
4. The top ImageNet class is shown as the result.
5. The result is appended to `results.csv`.

## Notes

- On the first run, TensorFlow/Keras may download pretrained model weights from the internet.
- Because the app uses pretrained ImageNet labels, predictions are limited to classes known by ImageNet.
- `vg16.py` is included for experimentation, but the main user flow runs through `app.py`.
- Some editor warnings may still appear in the codebase; they do not prevent the app from running.

## Author

Final Year Project prototype for experimenting with animal image recognition using pretrained deep learning models.
