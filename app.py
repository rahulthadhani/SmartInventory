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
from llm.generator import generate_description
from preprocessing.preprocess import preprocess_for_barcode, preprocess_for_ocr
from barcode.scanner import scan_barcode
import cv2
import threading
import base64
import numpy as np

app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")

import os

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


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def home():
    return render_template("home.html", total=0, today=0, recent=[])
    # products = get_all_products()
    # today_count = sum(
    #    1
    #    for p in products
    #    if p["timestamp"]
    #    and p["timestamp"].startswith(__import__("datetime").date.today().isoformat())
    # )
    # return render_template(
    #    "home.html", total=len(products), today=today_count, recent=products[:3]
    # )


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


@app.route("/api/lookup/<barcode>")
def lookup_barcode(barcode):
    """Looks up a single barcode in the database and returns the result."""
    existing = find_product_by_barcode(barcode.strip())
    return jsonify({"in_database": existing is not None, "product": existing})


# ── API Endpoints ─────────────────────────────────────────────────────────────


@app.route("/api/capture", methods=["POST"])
def capture():
    """
    Captures a frame from the camera, runs barcode detection,
    and returns all detected barcodes for the user to choose from.
    """
    with camera_lock:
        cap = get_camera()
        ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    # Try raw frame first then preprocessed
    results = scan_barcode(frame)
    if not results:
        processed = preprocess_for_barcode(frame)
        results = scan_barcode(processed)

    if not results:
        return jsonify({"found": False})

    if len(results) == 1:
        # Only one barcode found — proceed automatically
        barcode_value = results[0]["value"].strip()
        existing = find_product_by_barcode(barcode_value)
        return jsonify(
            {
                "found": True,
                "multiple": False,
                "barcode": barcode_value,
                "in_database": existing is not None,
                "product": existing,
            }
        )

    # Multiple barcodes found — return all for user to choose
    barcodes = []
    for r in results:
        value = r["value"].strip()
        existing = find_product_by_barcode(value)
        barcodes.append(
            {"value": value, "type": r["type"], "in_database": existing is not None}
        )

    return jsonify({"found": True, "multiple": True, "barcodes": barcodes})


@app.route("/api/ocr", methods=["POST"])
def run_ocr():
    """
    Captures a frame, runs OCR on it, sends results to LLM,
    and returns the generated product data as JSON.
    """
    with camera_lock:
        cap = get_camera()
        ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    barcode = request.json.get("barcode", "")

    ocr_frame = preprocess_for_ocr(frame)
    ocr_text = extract_text(ocr_frame)

    if not ocr_text or len(ocr_text) < 20:
        return jsonify(
            {
                "success": False,
                "message": "Not enough text detected. Try a different angle.",
            }
        )

    attributes = extract_product_attributes(ocr_text)
    llm_result = generate_description(barcode, attributes)

    if not llm_result:
        return jsonify({"success": False, "message": "LLM generation failed."})

    product_data = {
        "barcode": barcode,
        "brand": llm_result["brand"],
        "product_name": llm_result["product_name"],
        "product_type": llm_result["product_type"],
        "size": llm_result["size"],
        "ocr_text": ocr_text,
        "description": llm_result["description"],
    }

    validations = run_all_validations(product_data, find_product_by_barcode)

    return jsonify(
        {
            "success": True,
            "product": product_data,
            "validations": validations,
            "ocr_text": ocr_text,
        }
    )


@app.route("/api/save", methods=["POST"])
def save_product():
    """Saves a confirmed product record to the database."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    validations = run_all_validations(data, find_product_by_barcode)

    if not validations["is_valid"]:
        return jsonify({"error": "Validation failed", "validations": validations}), 400

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


@app.route("/api/products")
def get_products():
    """Returns all products as JSON for dynamic inventory updates."""
    products = get_all_products()
    return jsonify(products)


@app.route("/video_feed")
def video_feed():
    """
    Streams the live camera feed as MJPEG to the browser.
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


# ── Main ──────────────────────────────────────────────────────────────────────
@app.route("/test")
def test():
    return "<h1 style='color:white;background:black;padding:20px'>Flask is working</h1>"


if __name__ == "__main__":
    initialize_database()
    print("SmartInventory running at http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)
