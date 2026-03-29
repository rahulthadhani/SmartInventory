import easyocr
import re

# Initialize the EasyOCR reader once at module level
# This avoids reloading the model every time you call extract_text()
# gpu=False means it runs on CPU — set to True if you have a GPU
reader = easyocr.Reader(["en"], gpu=False)


def extract_text(frame):
    """
    Runs EasyOCR on a camera frame and returns all detected text
    as a single cleaned string.

    EasyOCR returns a list of tuples:
      (bounding_box, text, confidence_score)

    We only keep results above a confidence threshold of 0.3
    to filter out low quality detections.
    """
    results = reader.readtext(frame)

    if not results:
        return ""

    # Filter by confidence and extract just the text strings
    texts = [text for (_, text, confidence) in results if confidence > 0.3]

    # Join all detected text into one string
    raw_text = " ".join(texts)

    return clean_ocr_text(raw_text)


def clean_ocr_text(raw_text):
    """
    Cleans up raw OCR output by removing noise characters
    and normalizing whitespace.

    Common OCR artifacts:
    - Random symbols like |, @, #, ~
    - Extra spaces and newlines
    - Very short single character fragments
    """
    # Remove characters that are almost never meaningful in product text
    cleaned = re.sub(r"[|@#~^*<>{}[\]\\]", "", raw_text)

    # Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def extract_product_attributes(ocr_text):
    """
    Applies rule-based filtering to identify likely product
    attributes from raw OCR text.

    Looks for:
    - Brand/product name: capitalized words or ALL CAPS words
    - Size: patterns like "12 fl oz", "500ml", "2 lbs"
    - Keywords: common product type words

    Returns a dict of extracted attributes.
    These are best-effort guesses — the LLM will interpret them.
    """
    attributes = {
        "possible_brand": None,
        "possible_size": None,
        "possible_keywords": [],
        "raw_text": ocr_text,
    }

    # Look for size patterns e.g. 12 fl oz, 500ml, 2.5 lbs, 16oz
    size_pattern = re.search(
        r"\d+(\.\d+)?\s*(fl\.?\s*oz|ml|g|kg|lbs?|oz|L|liters?)", ocr_text, re.IGNORECASE
    )
    if size_pattern:
        attributes["possible_size"] = size_pattern.group().strip()

    # Look for ALL CAPS words as likely brand names (min 3 chars)
    caps_words = re.findall(r"\b[A-Z]{3,}\b", ocr_text)
    if caps_words:
        attributes["possible_brand"] = caps_words[0]

    # Collect meaningful keywords (words longer than 3 characters)
    # that might indicate product type
    stopwords = {
        "with",
        "and",
        "the",
        "for",
        "from",
        "this",
        "that",
        "are",
        "not",
        "per",
        "use",
        "may",
        "also",
    }
    words = ocr_text.split()
    keywords = [
        w.strip(".,;:") for w in words if len(w) > 3 and w.lower() not in stopwords
    ]
    attributes["possible_keywords"] = keywords[:10]  # cap at 10

    return attributes
