from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def build_prompt(barcode, attributes):
    """
    Builds the prompt sent to the LLM.

    The prompt is structured to:
    - Give the LLM only the facts we have
    - Explicitly forbid it from inventing details
    - Ask for a short, catalog-style description
    - Request a best guess at brand, product name, and type
      so we have structured fields to save alongside the description
    """
    ocr_text = attributes.get("raw_text", "")
    possible_brand = attributes.get("possible_brand", "Unknown")
    possible_size = attributes.get("possible_size", "Unknown")
    keywords = ", ".join(attributes.get("possible_keywords", []))

    prompt = f"""You are a product catalog assistant. Based only on the following 
information extracted from a product's packaging, generate a short product description 
and identify key product attributes.

Do NOT invent or assume any details that are not present in the provided text.
If information is missing, say "Unknown" for that field.

Barcode: {barcode}
Possible brand (from OCR): {possible_brand}
Possible size (from OCR): {possible_size}
Keywords found on packaging: {keywords}
Full OCR text from packaging: {ocr_text}

Respond in exactly this format:
Brand: <brand name or Unknown>
Product Name: <product name or Unknown>
Product Type: <product type/category or Unknown>
Size: <size or Unknown>
Description: <2-3 sentence catalog description using only the provided information>"""

    return prompt


def generate_description(barcode, attributes):
    """
    Sends the product attributes to the OpenAI API and returns
    the structured response as a parsed dictionary.

    Returns a dict with keys:
      brand, product_name, product_type, size, description

    Returns None if the API call fails.
    """
    prompt = build_prompt(barcode, attributes)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful product catalog assistant. "
                    "Only use provided information. Never invent details.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,  # low temperature = more factual, less creative
        )

        raw_response = response.choices[0].message.content.strip()
        return parse_llm_response(raw_response)

    except Exception as e:
        print(f"LLM API error: {e}")
        return None


def parse_llm_response(raw_response):
    """
    Parses the structured LLM response into a dictionary.

    Expected format:
      Brand: Celsius
      Product Name: Sparkling Orange Energy Drink
      Product Type: Energy Drink
      Size: 12 fl oz
      Description: Celsius Sparkling Orange is a...

    Returns a dict with those five keys.
    Falls back to "Unknown" for any field that couldn't be parsed.
    """
    result = {
        "brand": "Unknown",
        "product_name": "Unknown",
        "product_type": "Unknown",
        "size": "Unknown",
        "description": "Unknown",
    }

    lines = raw_response.strip().split("\n")
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key in result:
            result[key] = value

    return result
