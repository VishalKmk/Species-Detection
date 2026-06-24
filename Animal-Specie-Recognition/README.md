# Animal Species Recognition

An image-based animal classification tool built with a Flask REST API backend and a plain HTML/CSS/JS frontend. Uses pretrained TensorFlow/Keras models (ResNet50 and VGG16) with ImageNet weights.

## Project Structure

```text
Animal-Specie-Recognition/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── requirements.txt    # Python dependencies
│   ├── vg16.py             # VGG16 experiment module (legacy)
│   ├── test.ipynb
│   ├── testmodel.ipynb
│   └── venv/               # (gitignored)
├── frontend/
│   └── index.html          # UI — no build step, no framework
├── animals/                # Sample test images
└── results.csv             # Prediction log (gitignored)
```

## Tech Stack

- Python `3.12`
- Flask + flask-cors
- TensorFlow / Keras (ResNet50, VGG16)
- NumPy, Pillow
- HTML / CSS / JS (vanilla, no framework)

## Setup

### 1. Create and activate a virtual environment

```powershell
cd backend
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the backend

```powershell
python app.py
```

Flask starts on `http://localhost:5000`. Both models load at startup — first run will download ImageNet weights.

### 4. Serve the frontend

Open a second terminal:

```powershell
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Accepts `image` (file) and `model` (`ResNet50` or `VGG16`) as form-data. Returns `species`, `accuracy`, `model`. |
| GET | `/api/history` | Returns last 20 predictions as JSON. |
| GET | `/api/history/download` | Downloads `results.csv`. |
| GET | `/api/health` | Health check. |

## How It Works

1. Upload an image (JPG or PNG) via the browser UI.
2. Select a model — ResNet50 or VGG16.
3. The frontend POSTs the image to `/api/predict`.
4. Flask resizes it to `224×224`, runs inference, and returns the top ImageNet class with confidence.
5. The result is logged to `results.csv` and displayed in the UI.

## Notes

- Predictions are limited to ImageNet classes — this is a pretrained classification demo, not a custom-trained species pipeline.
- Both models load once at startup, not per request.
- The frontend runs on port `8000`, the backend on port `5000`. CORS is enabled so they can communicate.