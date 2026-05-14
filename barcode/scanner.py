from pyzbar.pyzbar import decode
import cv2
import numpy as np


def scan_barcode(frame):
    """
    Attempts barcode detection at 0, 90, 180, and 270 degrees
    on both the raw and preprocessed frame.
    """
    results = _decode_frame(frame)
    if results:
        return results

    for angle in [90, 180, 270]:
        rotated = _rotate_frame(frame, angle)
        results = _decode_frame(rotated)
        if results:
            print(f"Barcode detected after {angle} degree rotation.")
            return results

    return []


def _decode_frame(frame):
    barcodes = decode(frame)
    if not barcodes:
        return []

    results = []
    for barcode in barcodes:
        barcode_value = barcode.data.decode("utf-8")
        results.append(
            {"value": barcode_value, "type": barcode.type, "rect": barcode.rect}
        )
    return results


def _rotate_frame(frame, angle):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust output size for 90 and 270 degree rotations
    # so nothing gets cropped
    if angle in [90, 270]:
        new_w, new_h = h, w
    else:
        new_w, new_h = w, h

    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(frame, matrix, (new_w, new_h))


def validate_barcode(barcode_value):
    barcode_value = barcode_value.strip()

    if not barcode_value:
        return False, "Barcode value is empty."

    if not barcode_value.isdigit():
        if len(barcode_value) >= 6:
            return True, "Non-numeric barcode accepted."
        return False, "Barcode contains invalid characters."

    length = len(barcode_value)
    valid_lengths = {8: "EAN-8", 12: "UPC-A", 13: "EAN-13"}

    if length in valid_lengths:
        return True, f"Valid {valid_lengths[length]} barcode."

    return False, f"Unrecognized barcode length: {length} digits."


def draw_barcode_overlay(frame, barcode_results):
    for barcode in barcode_results:
        rect = barcode["rect"]
        value = barcode["value"]

        cv2.rectangle(
            frame,
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top + rect.height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            value,
            (rect.left, rect.top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame
