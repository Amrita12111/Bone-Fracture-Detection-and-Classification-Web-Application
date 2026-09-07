from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.inference import FractureDetector
import os, time
 
app = Flask(__name__)
CORS(app)
 
BASE         = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE, "model", "weights", "best.pt")
TYPE_MODEL   = os.path.join(BASE, "model", "weights", "best_type_classifier.pt")
BODY_MODEL   = os.path.join(BASE, "model", "weights", "best_body_part_classifier.pt")   # ← new

detector = FractureDetector(MODEL_PATH, type_model_path=TYPE_MODEL, body_part_model_path=BODY_MODEL)
 
ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_MB  = 16 * 1024 * 1024
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","model":"YOLOv8m-FractureDetector",
                    "type_model": os.path.exists(TYPE_MODEL)})
 
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success":False,"error":"No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename or "." not in file.filename:
        return jsonify({"success":False,"error":"Invalid filename"}), 400
    if file.filename.rsplit(".",1)[1].lower() not in ALLOWED:
        return jsonify({"success":False,"error":"File type not allowed"}), 400
 
    image_bytes = file.read()
    if len(image_bytes) > MAX_MB:
        return jsonify({"success":False,"error":"File too large (max 16MB)"}), 413
 
    conf   = float(request.form.get("confidence", 0.25))
    conf   = max(0.1, min(0.9, conf))
    start  = time.time()
    result = detector.detect(image_bytes, conf_threshold=conf)
    result["inference_time_ms"] = round((time.time()-start)*1000, 1)
    result["filename"]          = file.filename
    return jsonify(result)
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
 