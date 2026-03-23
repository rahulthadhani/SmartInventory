import cv2
import numpy as np
import os


def to_grayscale(frame):
    """
    Converts a color image (BGR) to grayscale.
    Grayscale removes color noise and is required by both
    Pyzbar (barcode) and EasyOCR for better accuracy.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray_frame):
    """
    Applies a Gaussian blur to smooth out image noise.
    The (5, 5) kernel size controls how strong the blur is.
    Larger = more blur. Don't go above (9,9) or you'll lose detail.
    """
    return cv2.GaussianBlur(gray_frame, (5, 5), 0)


def increase_contrast(gray_frame):
    """
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast in dark or unevenly lit images.
    This helps barcodes show up more clearly under bad lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)


def threshold_image(gray_frame):
    """
    Converts the grayscale image to pure black and white.
    Uses Otsu's method to automatically find the best threshold value.
    This makes barcodes and text edges sharper and easier to detect.
    """
    _, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def crop_center(frame, crop_ratio=0.7):
    """
    Crops the center portion of the frame.
    Products are usually held in the center of the camera view,
    so cropping removes background clutter.
    crop_ratio=0.7 keeps the center 70% of the image.
    """
    h, w = frame.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return frame[margin_y : h - margin_y, margin_x : w - margin_x]


def preprocess_for_barcode(frame):
    """
    Full preprocessing pipeline tuned for barcode detection.
    Barcodes need sharp, high-contrast black and white lines.
    Steps: grayscale → noise reduction → contrast boost → threshold
    Returns the processed image as a NumPy array.
    """
    gray = to_grayscale(frame)
    blurred = reduce_noise(gray)
    contrasted = increase_contrast(blurred)
    thresh = threshold_image(contrasted)
    return thresh


def preprocess_for_ocr(frame):
    """
    Full preprocessing pipeline tuned for OCR text extraction.
    OCR works better with clear contrast but doesn't need
    full thresholding — grayscale + contrast is usually enough.
    Returns the processed image as a NumPy array.
    """
    gray = to_grayscale(frame)
    contrasted = increase_contrast(gray)
    return contrasted


def save_preprocessed(image, filename, output_dir="data/sample_frames"):
    """
    Saves a preprocessed (grayscale) image to disk.
    Used to inspect your preprocessing results visually.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Preprocessed image saved to {filepath}")
    return filepath
