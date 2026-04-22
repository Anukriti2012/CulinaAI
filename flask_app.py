from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "page1.html")

@app.route("/predict-image", methods=["POST"])
def predict_image():
    files = {"image": (request.files["image"].filename, request.files["image"].stream, request.files["image"].mimetype)}
    response = requests.post(f"{FASTAPI_BASE_URL}/predict-image", files=files)
    return jsonify(response.json())

@app.route("/rerank", methods=["POST"])
def rerank():
    response = requests.post(f"{FASTAPI_BASE_URL}/rerank", json=request.get_json())
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(port=5000, debug=True)