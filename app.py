from flask import Flask, render_template, jsonify, request, Response
from database.db import initialize_database
from database.queries import (
    find_product_by_barcode,
    insert_product,
    update_product,
    get_all_products,
    delete_product,
)
from validation.validator import run_all_validations
from ocr.extractor import extract_text, extract_product_attributes
from llm.generator import generate_description, find_product_image
from preprocessing.preprocess import preprocess_for_barcode, preprocess_for_ocr
from barcode.scanner import scan_barcode
import cv2
import threading
import base64
import numpy as np
import os

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")

print("Template folder:", os.path.abspath(app.template_folder))
print("Static folder:", os.path.abspath(app.static_folder))
print("Templates found:", os.listdir(app.template_folder))

# Global camera state
camera = None
camera_lock = threading.Lock()


def get_camera():
    """Returns the global camera instance, opening it if needed."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def release_camera():
    """Releases the global camera instance."""
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None


def resize_if_needed(frame, max_side=1600):
    """
    Resizes a frame if its longest side exceeds max_side.
    Prevents EasyOCR from freezing on large iPhone photos (4032x3024).
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from ({w}x{h}) to ({new_w}x{new_h})")
    return frame


def decode_image_from_request(file):
    """
    Reads image bytes from a Flask file object and decodes to a cv2 frame.
    Returns (frame, error_message) — frame is None if decoding failed.
    """
    file_bytes = file.read()
    if not file_bytes:
        return None, "Empty image received"

    img_array = np.frombuffer(file_bytes, dtype=np.uint8)
    if img_array.size == 0:
        return None, "Could not read image bytes"

    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "Could not decode image"

    return frame, None


# ── Page Routes ───────────────────────────────────────────────────────────────


@app.route("/")
def home():
    products = get_all_products()
    today_count = sum(
        1
        for p in products
        if p["timestamp"]
        and p["timestamp"].startswith(__import__("datetime").date.today().isoformat())
    )
    return render_template(
        "home.html", total=len(products), today=today_count, recent=products[:3]
    )


@app.route("/scan")
def scan():
    return render_template("scan.html")


@app.route("/inventory")
def inventory():
    products = get_all_products()
    return render_template("inventory.html", products=products)


@app.route("/review/<barcode>")
def review(barcode):
    product = find_product_by_barcode(barcode)
    if not product:
        return render_template("review.html", product=None, barcode=barcode)
    validations = run_all_validations(product, find_product_by_barcode)
    return render_template(
        "review.html", product=product, validations=validations, barcode=barcode
    )


@app.route("/test")
def test():
    return "<h1 style='color:white;background:black;padding:20px'>Flask is working</h1>"


# ── Web UI API Endpoints ──────────────────────────────────────────────────────


@app.route("/api/lookup/<barcode>")
def lookup_barcode(barcode):
    """Looks up a single barcode in the database and returns the result."""
    existing = find_product_by_barcode(barcode.strip())
    return jsonify({"in_database": existing is not None, "product": existing})


@app.route("/api/products")
def get_products():
    """Returns all products as JSON for dynamic inventory updates."""
    products = get_all_products()
    return jsonify(products)


@app.route("/api/capture", methods=["POST"])
def capture():
    """
    Captures a frame from the webcam, runs barcode detection,
    and returns all detected barcodes for the user to choose from.
    """
    with camera_lock:
        cap = get_camera()
        ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    results = scan_barcode(frame)
    if not results:
        processed = preprocess_for_barcode(frame)
        results = scan_barcode(processed)

    if not results:
        return jsonify({"found": False})

    if len(results) == 1:
        barcode_value = results[0]["value"].strip()
        existing = find_product_by_barcode(barcode_value)
        return jsonify({
            "found":       True,
            "multiple":    False,
            "barcode":     barcode_value,
            "in_database": existing is not None,
            "product":     existing,
        })

    barcodes = []
    for r in results:
        value = r["value"].strip()
        existing = find_product_by_barcode(value)
        barcodes.append({
            "value":       value,
            "type":        r["type"],
            "in_database": existing is not None
        })

    return jsonify({"found": True, "multiple": True, "barcodes": barcodes})


@app.route("/api/capture_frame", methods=["POST"])
def capture_frame_image():
    """
    Captures the current webcam frame and returns it
    as a base64 encoded JPEG for display in the browser.
    """
    with camera_lock:
        cap = get_camera()
        ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"image": f"data:image/jpeg;base64,{frame_b64}"})


@app.route("/api/ocr", methods=["POST"])
def run_ocr():
    """
    Captures a frame from the webcam, runs OCR and LLM description generation,
    and searches for a product image online.
    Used by the web UI scan page.
    """
    with camera_lock:
        cap = get_camera()
        ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    barcode = request.json.get("barcode", "")

    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    frame_data_url = f"data:image/jpeg;base64,{frame_b64}"

    ocr_frames = preprocess_for_ocr(frame)
    ocr_text = extract_text(ocr_frames)

    if not ocr_text or len(ocr_text) < 20:
        return jsonify({
            "success": False,
            "message": "Not enough text detected. Try holding the product closer and showing the full front label.",
            "frame":   frame_data_url,
        })

    attributes = extract_product_attributes(ocr_text)
    llm_result = generate_description(barcode, attributes)

    if not llm_result:
        return jsonify({
            "success": False,
            "message": "LLM generation failed.",
            "frame":   frame_data_url,
        })

    image_url = find_product_image(
        llm_result.get("product_name", ""),
        llm_result.get("brand", ""),
        barcode
    )

    product_data = {
        "barcode":      barcode,
        "brand":        llm_result["brand"],
        "product_name": llm_result["product_name"],
        "product_type": llm_result["product_type"],
        "size":         llm_result["size"],
        "ocr_text":     ocr_text,
        "description":  llm_result["description"],
        "image_url":    image_url,
    }

    validations = run_all_validations(product_data, find_product_by_barcode)

    return jsonify({
        "success":     True,
        "product":     product_data,
        "validations": validations,
        "ocr_text":    ocr_text,
        "frame":       frame_data_url,
        "image_url":   image_url,
    })


@app.route("/api/upload_scan", methods=["POST"])
def upload_scan():
    """
    Accepts an uploaded image from the web UI,
    runs barcode detection and returns the result.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    frame, error = decode_image_from_request(file)
    if frame is None:
        return jsonify({"error": error}), 400

    frame = resize_if_needed(frame)

    results = scan_barcode(frame)
    if not results:
        processed = preprocess_for_barcode(frame)
        results = scan_barcode(processed)

    if not results:
        return jsonify({"found": False, "message": "No barcode detected in image."})

    if len(results) == 1:
        barcode_value = results[0]["value"].strip()
        existing = find_product_by_barcode(barcode_value)
        return jsonify({
            "found":       True,
            "multiple":    False,
            "barcode":     barcode_value,
            "in_database": existing is not None,
            "product":     existing,
        })

    barcodes = []
    for r in results:
        value = r["value"].strip()
        existing = find_product_by_barcode(value)
        barcodes.append({
            "value":       value,
            "type":        r["type"],
            "in_database": existing is not None
        })

    return jsonify({"found": True, "multiple": True, "barcodes": barcodes})


@app.route("/api/upload_ocr", methods=["POST"])
def upload_ocr():
    """
    Accepts an uploaded image from the web UI,
    runs OCR and LLM description generation,
    and searches for a product image online.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    barcode = request.form.get("barcode", "")

    frame, error = decode_image_from_request(file)
    if frame is None:
        return jsonify({"success": False, "message": error}), 400

    frame = resize_if_needed(frame)

    ocr_frames = preprocess_for_ocr(frame)
    ocr_text = extract_text(ocr_frames)

    if not ocr_text or len(ocr_text) < 20:
        return jsonify({
            "success": False,
            "message": "Not enough text detected. Try a clearer image showing the full product front.",
        })

    attributes = extract_product_attributes(ocr_text)
    llm_result = generate_description(barcode, attributes)

    if not llm_result:
        return jsonify({"success": False, "message": "LLM generation failed."})

    image_url = find_product_image(
        llm_result.get("product_name", ""),
        llm_result.get("brand", ""),
        barcode
    )

    product_data = {
        "barcode":      barcode,
        "brand":        llm_result["brand"],
        "product_name": llm_result["product_name"],
        "product_type": llm_result["product_type"],
        "size":         llm_result["size"],
        "ocr_text":     ocr_text,
        "description":  llm_result["description"],
        "image_url":    image_url,
    }

    validations = run_all_validations(product_data, find_product_by_barcode)

    return jsonify({
        "success":     True,
        "product":     product_data,
        "validations": validations,
        "ocr_text":    ocr_text,
        "image_url":   image_url,
    })


@app.route("/api/save", methods=["POST"])
def save_product():
    """Saves a confirmed product record to the database."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Check for duplicate before attempting insert
    existing = find_product_by_barcode(data.get("barcode", ""))
    if existing:
        return jsonify({
            "success":        True,
            "message":        "Product already exists in database.",
            "id":             existing.get("id"),
            "already_exists": True
        })

    validations = run_all_validations(data, find_product_by_barcode)
    if not validations["is_valid"]:
        return jsonify({"error": "Validation failed", "validations": validations}), 400

    product_id = insert_product(data)
    if product_id:
        return jsonify({"success": True, "id": product_id})

    return jsonify({"error": "Failed to save product"}), 500


@app.route("/api/save_manual", methods=["POST"])
def save_manual():
    """Saves a manually entered product record, skipping strict validation."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if not data.get("barcode"):
        return jsonify({"error": "Barcode is required"}), 400

    existing = find_product_by_barcode(data["barcode"])
    if existing:
        return jsonify({"error": "Product with this barcode already exists"}), 400

    product_id = insert_product(data)
    if product_id:
        return jsonify({"success": True, "id": product_id})

    return jsonify({"error": "Failed to save product"}), 500


@app.route("/api/update/<barcode>", methods=["POST"])
def update_product_route(barcode):
    """Updates an existing product record."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    update_product(barcode, data)
    return jsonify({"success": True})


@app.route("/api/delete/<barcode>", methods=["DELETE"])
def delete_product_route(barcode):
    """Deletes a product record by barcode."""
    delete_product(barcode)
    return jsonify({"success": True})


@app.route("/video_feed")
def video_feed():
    """
    Streams the live webcam feed as MJPEG to the browser.
    This powers the live camera view on the scan page.
    """
    def generate_frames():
        while True:
            with camera_lock:
                cap = get_camera()
                ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── Mobile API Endpoints ──────────────────────────────────────────────────────


@app.route("/api/scan_image", methods=["POST"])
def scan_image():
    """
    Accepts an uploaded image from the mobile app,
    runs barcode detection and returns the result.
    Handles large iPhone images (4032x3024) by resizing before processing.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    frame, error = decode_image_from_request(file)
    if frame is None:
        return jsonify({"found": False, "error": error}), 400

    frame = resize_if_needed(frame)

    results = scan_barcode(frame)
    if not results:
        processed = preprocess_for_barcode(frame)
        results = scan_barcode(processed)

    if not results:
        return jsonify({"found": False})

    if len(results) == 1:
        barcode_value = results[0]["value"].strip()
        existing = find_product_by_barcode(barcode_value)
        return jsonify({
            "found":       True,
            "multiple":    False,
            "barcode":     barcode_value,
            "in_database": existing is not None,
            "product":     existing,
        })

    barcodes = []
    for r in results:
        value = r["value"].strip()
        existing = find_product_by_barcode(value)
        barcodes.append({
            "value":       value,
            "type":        r["type"],
            "in_database": existing is not None
        })

    return jsonify({"found": True, "multiple": True, "barcodes": barcodes})


@app.route("/api/ocr_image", methods=["POST"])
def ocr_image():
    """
    Accepts an uploaded image from the mobile app,
    runs OCR and LLM description generation,
    and searches for a product image online.
    Handles large iPhone images by resizing before OCR to prevent freezing.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    barcode = request.form.get("barcode", "")

    frame, error = decode_image_from_request(file)
    if frame is None:
        return jsonify({"success": False, "message": error}), 400

    frame = resize_if_needed(frame)

    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    frame_data_url = f"data:image/jpeg;base64,{frame_b64}"

    ocr_frames = preprocess_for_ocr(frame)
    ocr_text = extract_text(ocr_frames)

    if not ocr_text or len(ocr_text) < 20:
        return jsonify({
            "success": False,
            "message": "Not enough text detected. Try holding the product closer and showing the full front label.",
            "frame":   frame_data_url,
        })

    attributes = extract_product_attributes(ocr_text)
    llm_result = generate_description(barcode, attributes)

    if not llm_result:
        return jsonify({
            "success": False,
            "message": "LLM generation failed.",
            "frame":   frame_data_url,
        })

    image_url = find_product_image(
        llm_result.get("product_name", ""),
        llm_result.get("brand", ""),
        barcode
    )

    product_data = {
        "barcode":      barcode,
        "brand":        llm_result["brand"],
        "product_name": llm_result["product_name"],
        "product_type": llm_result["product_type"],
        "size":         llm_result["size"],
        "ocr_text":     ocr_text,
        "description":  llm_result["description"],
        "image_url":    image_url,
    }

    validations = run_all_validations(product_data, find_product_by_barcode)

    return jsonify({
        "success":     True,
        "product":     product_data,
        "validations": validations,
        "ocr_text":    ocr_text,
        "frame":       frame_data_url,
        "image_url":   image_url,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialize_database()
    print("SmartInventory running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)