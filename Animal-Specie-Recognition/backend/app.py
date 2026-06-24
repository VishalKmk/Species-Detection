"""
Flask API backend for Animal Species Recognition.

Replaces the Streamlit UI with a plain REST API so any frontend
(HTML/JS, React, mobile, etc.) can call it.

Run with:
    python app.py
Then it listens on http://localhost:5000
"""

import csv
import io
import os
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input as resnet_preprocess,
    decode_predictions as resnet_decode,
)
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg_preprocess,
    decode_predictions as vgg_decode,
)

app = Flask(__name__)
CORS(app)  # allow the static frontend (different origin/port) to call this API

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.csv")

# ---------------------------------------------------------------------------
# Load both models ONCE at startup, not per-request.
# The original app.py called ResNet50(weights='imagenet') inside the predict
# function, which reloads ~100MB of weights on every single click. This was
# the single biggest performance problem in the original code.
# ---------------------------------------------------------------------------
print("Loading ResNet50...")
resnet_model = ResNet50(weights="imagenet")
print("Loading VGG16...")
vgg_model = VGG16(weights="imagenet")
print("Models ready.")


def preprocess_image(file_storage, target_size=(224, 224)):
    """Read an uploaded file into a (1, 224, 224, 3) array, model-agnostic."""
    img = Image.open(file_storage.stream).convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_resnet(arr):
    x = resnet_preprocess(arr.copy())
    preds = resnet_model.predict(x, verbose=0)
    label, name, score = resnet_decode(preds, top=1)[0][0]
    return name.replace("_", " "), float(score)


def predict_vgg16(arr):
    x = vgg_preprocess(arr.copy())
    preds = vgg_model.predict(x, verbose=0)
    label, name, score = vgg_decode(preds, top=1)[0][0]
    return name.replace("_", " "), float(score)


def save_result(species, accuracy, model_name):
    now = datetime.now()
    is_new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["Date", "Time", "Species", "Accuracy", "Model"])
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            species,
            accuracy,
            model_name,
        ])


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected field 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    model_choice = request.form.get("model", "ResNet50")

    try:
        arr = preprocess_image(file)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    if model_choice == "VGG16":
        species, accuracy = predict_vgg16(arr)
    else:
        species, accuracy = predict_resnet(arr)
        model_choice = "ResNet50"

    save_result(species, accuracy, model_choice)

    return jsonify({
        "species": species,
        "accuracy": accuracy,
        "model": model_choice,
    })


@app.route("/api/history", methods=["GET"])
def history():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"results": []})

    rows = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # most recent first
    return jsonify({"results": list(reversed(rows))})


@app.route("/api/history/download", methods=["GET"])
def download_history():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results yet"}), 404
    return send_file(RESULTS_FILE, as_attachment=True, download_name="results.csv")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)