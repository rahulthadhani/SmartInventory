import cv2


def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray_frame):
    return cv2.GaussianBlur(gray_frame, (5, 5), 0)


def increase_contrast(gray_frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)


def threshold_image(gray_frame):
    _, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def crop_center(frame, crop_ratio=0.7):
    h, w = frame.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return frame[margin_y : h - margin_y, margin_x : w - margin_x]


def rotate_frame(frame, angle):
    """
    Rotates a frame by the given angle.
    Adjusts output dimensions for 90 and 270 degree rotations
    so no content is cropped.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if angle in [90, 270]:
        new_w, new_h = h, w
    else:
        new_w, new_h = w, h

    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(frame, matrix, (new_w, new_h))


def preprocess_for_barcode(frame):
    """
    Preprocessing pipeline for barcode detection.
    Returns a single processed frame.
    """
    gray = to_grayscale(frame)
    blurred = reduce_noise(gray)
    contrasted = increase_contrast(blurred)
    thresh = threshold_image(contrasted)
    return thresh


def preprocess_for_ocr(frame):
    """
    Returns color frame variants for PaddleOCR.
    PaddleOCR v3 runs its own internal preprocessing, so we pass BGR color
    frames rather than grayscale. Upscaling helps with small label text;
    the 90-degree rotation catches sideways text.

    Versions returned:
    1. Original color frame
    2. 2x upscaled color frame (for small text)
    3. 90 degree rotation of original (for vertical text)
    """
    variants = []
    variants.append(frame.copy())
    variants.append(cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    variants.append(rotate_frame(frame, 90))
    return variants
