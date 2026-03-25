from pyzbar.pyzbar import decode
import cv2


def scan_barcode(frame):
    """
    Takes a raw or preprocessed frame and attempts to detect
    and decode any barcodes present in the image.

    pyzbar.decode() returns a list of detected barcode objects.
    Each object has:
      - data:   the raw barcode bytes (we decode to a string)
      - type:   the barcode format e.g. EAN13, UPCA, CODE128
      - rect:   the bounding box of the barcode in the image

    Returns a list of dicts, one per detected barcode.
    Returns an empty list if nothing is found.
    """
    barcodes = decode(frame)

    if not barcodes:
        return []

    results = []
    for barcode in barcodes:
        barcode_value = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        results.append(
            {"value": barcode_value, "type": barcode_type, "rect": barcode.rect}
        )

    return results


def validate_barcode(barcode_value):
    # Strip whitespace before validating
    barcode_value = barcode_value.strip()

    if not barcode_value:
        return False, "Barcode value is empty"

    if not barcode_value.isdigit():
        if len(barcode_value) >= 6:
            return True, "Non-numeric barcode (CODE128 or similar) — accepted"
        return False, "Barcode contains invalid characters"

    length = len(barcode_value)
    valid_lengths = {8: "EAN-8", 12: "UPC-A", 13: "EAN-13"}

    if length in valid_lengths:
        return True, f"Valid {valid_lengths[length]} barcode"

    return False, f"Unrecognized barcode length: {length} digits"


def draw_barcode_overlay(frame, barcode_results):
    """
    Draws a green rectangle and the decoded value on the frame
    around each detected barcode. Useful for visual debugging.

    Returns the annotated frame.
    """
    for barcode in barcode_results:
        rect = barcode["rect"]
        value = barcode["value"]

        # Draw green bounding box
        cv2.rectangle(
            frame,
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top + rect.height),
            (0, 255, 0),
            2,
        )

        # Draw decoded value above the box
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
