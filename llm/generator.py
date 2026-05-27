import openai
import json
import re
from config import OPENAI_API_KEY, OPENAI_MODEL

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def generate_description(barcode, attributes):
    """
    Generates a structured product description using GPT-4o-mini with web search.
    Returns a dict with brand, product_name, product_type, size, and description.
    """
    raw_text = attributes.get("raw_text", "")
    possible_brand = attributes.get("possible_brand", "Unknown")
    possible_size = attributes.get("possible_size", "")
    keywords = " ".join(attributes.get("possible_keywords", []))

    prompt = f"""You are a product catalog assistant. Based on the following information extracted from a product label, generate accurate product details.

Barcode: {barcode}
Raw OCR text: {raw_text}
Possible brand: {possible_brand}
Possible size: {possible_size}
Keywords: {keywords}

Search the web to verify and correct any OCR errors. Return ONLY a valid JSON object with no extra text:

{{
  "brand": "exact brand name",
  "product_name": "full product name",
  "product_type": "category like Beverage, Snack, Electronics, etc",
  "size": "size or weight",
  "description": "2-3 sentence product description"
}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_preview"}]
        )

        content = response.choices[0].message.content
        if not content:
            for block in response.choices[0].message.tool_calls or []:
                if hasattr(block, "function"):
                    content = block.function.arguments
                    break

        if not content:
            return None

        clean = re.sub(r"```json|```", "", content).strip()
        return json.loads(clean)

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"LLM generation error: {e}")
        return None


def find_product_image(product_name, brand, barcode):
    """
    Uses GPT-4o-mini with web search to find a publicly accessible
    product image URL for the given product.
    Returns a URL string or None if not found.
    """
    prompt = f"""Search the web for a product image of:
Product: {product_name}
Brand: {brand}
Barcode: {barcode}

Find a direct image URL (ending in .jpg, .jpeg, .png, or .webp) from a 
reputable source like the brand's official website, Amazon, Walmart, 
Target, or a major retailer.

Return ONLY a JSON object with no extra text:
{{
  "image_url": "https://example.com/product-image.jpg",
  "source": "amazon.com"
}}

If no image is found return:
{{
  "image_url": null,
  "source": null
}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_preview"}]
        )

        content = response.choices[0].message.content
        if not content:
            return None

        clean = re.sub(r"```json|```", "", content).strip()
        result = json.loads(clean)
        image_url = result.get("image_url")

        if image_url and _is_valid_image_url(image_url):
            print(f"Found product image: {image_url}")
            return image_url

        return None

    except Exception as e:
        print(f"Image search error: {e}")
        return None


def _is_valid_image_url(url):
    """Basic check that the URL looks like a direct image link."""
    if not url or not url.startswith("http"):
        return False
    lower = url.lower()
    return any(lower.endswith(ext) for ext in
               [".jpg", ".jpeg", ".png", ".webp", ".gif"])