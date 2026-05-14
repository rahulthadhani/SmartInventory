from paddleocr import PaddleOCR
import re

# enable_mkldnn=False avoids a Windows oneDNN incompatibility in PaddleOCR v3.
# use_textline_orientation handles 0/180 degree text per detected line.
reader = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)


def extract_text(frames):
    """
    Runs PaddleOCR on multiple frame variants.
    Deduplicates results across all variants.
    """
    if not isinstance(frames, list):
        frames = [frames]

    all_texts = []
    seen = set()

    for i, frame in enumerate(frames):
        try:
            results = reader.predict(frame)
            for res in results:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                for text, confidence in zip(texts, scores):
                    text_clean = text.strip()
                    text_lower = text_clean.lower()
                    if confidence > 0.1 and text_lower not in seen and len(text_clean) > 1:
                        all_texts.append(text_clean)
                        seen.add(text_lower)
        except Exception as e:
            print(f"OCR error on frame variant {i}: {e}")
            continue

    raw_text = " ".join(all_texts)
    return clean_ocr_text(raw_text)


def clean_ocr_text(raw_text):
    cleaned = re.sub(r"[|@#~^*<>{}[\]\\]", "", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def extract_product_attributes(ocr_text):
    attributes = {
        "possible_brand": None,
        "possible_size": None,
        "possible_keywords": [],
        "raw_text": ocr_text,
    }

    size_pattern = re.search(
        r"\d+(\.\d+)?\s*(fl\.?\s*oz|ml|g|kg|lbs?|oz|L|liters?)", ocr_text, re.IGNORECASE
    )
    if size_pattern:
        attributes["possible_size"] = size_pattern.group().strip()

    caps_words = re.findall(r"\b[A-Z]{3,}\b", ocr_text)
    if caps_words:
        attributes["possible_brand"] = caps_words[0]

    stopwords = {
        "with", "and", "the", "for", "from", "this", "that",
        "are", "not", "per", "use", "may", "also",
    }
    words = ocr_text.split()
    keywords = [
        w.strip(".,;:") for w in words if len(w) > 3 and w.lower() not in stopwords
    ]
    attributes["possible_keywords"] = keywords[:10]

    return attributes
