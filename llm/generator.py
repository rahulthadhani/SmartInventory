from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def build_prompt(barcode, attributes):
    """
    Builds the prompt sent to the LLM.
    The LLM is instructed to use web search to verify and correct
    product information before generating the description.
    """
    ocr_text = attributes.get("raw_text", "")
    possible_brand = attributes.get("possible_brand", "Unknown")
    possible_size = attributes.get("possible_size", "Unknown")
    keywords = ", ".join(attributes.get("possible_keywords", []))

    prompt = f"""You are a product catalog assistant with access to web search.

A product was scanned and the following information was extracted from its packaging 
using OCR. OCR text is often noisy and may contain errors — for example brand names 
may be misspelled or garbled.

Your job is to:
1. Use the barcode and OCR text to search the web and identify the real product
2. Correct any OCR errors in the brand name, product name, and other fields
3. Use the verified real product information to fill in the fields accurately
4. Generate a short, accurate catalog description based on verified information

Barcode: {barcode}
Possible brand (from OCR, may contain errors): {possible_brand}
Possible size (from OCR): {possible_size}
Keywords found on packaging: {keywords}
Full OCR text from packaging (may contain errors): {ocr_text}

Search for this product using the barcode and OCR keywords to find the correct 
product details. Correct any OCR errors you find.

Respond in exactly this format:
Brand: <verified brand name>
Product Name: <verified product name>
Product Type: <verified product type/category>
Size: <verified size or Unknown>
Description: <2-3 sentence catalog description using verified information>"""

    return prompt


def generate_description(barcode, attributes):
    """
    Sends product attributes to the OpenAI API with web search enabled.
    The LLM will search the web to verify and correct OCR errors before
    generating the description.

    Returns a dict with keys:
      brand, product_name, product_type, size, description

    Returns None if the API call fails.
    """
    prompt = build_prompt(barcode, attributes)

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )

        # Extract the text output from the response
        raw_response = ""
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        raw_response += block.text

        raw_response = raw_response.strip()

        if not raw_response:
            print("LLM returned empty response.")
            return None

        print(f"\nLLM raw response:\n{raw_response}\n")
        return parse_llm_response(raw_response)

    except Exception as e:
        print(f"LLM API error: {e}")
        return None


def parse_llm_response(raw_response):
    """
    Parses the structured LLM response into a dictionary.

    Expected format:
      Brand: PowerA
      Product Name: ADVANTAGE Wired Controller for Xbox
      Product Type: Gaming Controller
      Size: Unknown
      Description: The PowerA ADVANTAGE Wired Controller...

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
