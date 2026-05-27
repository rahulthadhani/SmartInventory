import uuid
import re
from database.supabase_client import supabase
from config import DEFAULT_CATEGORY_ID


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_slug(title, sku):
    """
    Generates a URL-friendly slug from the product title.
    Appends last 4 digits of SKU to ensure uniqueness.
    Example: 'Celsius Sparkling Orange' -> 'celsius-sparkling-orange-4870'
    """
    base = title or sku
    slug = re.sub(r'[^a-z0-9]+', '-', base.lower()).strip('-')
    return f"{slug}-{sku[-4:]}"


def _extract_tag(tags, prefix):
    """
    Extracts a value from the tags array by prefix.
    Example: tags = ["brand:Celsius", "size:12 fl oz"]
             _extract_tag(tags, "brand") -> "Celsius"
    """
    if not tags:
        return None
    for tag in tags:
        if tag.startswith(f"{prefix}:"):
            return tag[len(prefix) + 1:]
    return None


def _get_raw_tags(barcode_value):
    """Returns the raw tags array for a product by SKU."""
    try:
        response = supabase.table("Product")\
            .select("tags")\
            .eq("sku", barcode_value)\
            .execute()
        if response.data:
            return response.data[0].get("tags") or []
    except Exception as e:
        print(f"Supabase error in _get_raw_tags: {e}")
    return []


def _update_tags(current_tags, updates):
    """
    Updates specific tag values while preserving others.
    updates = {"brand": "Celsius", "size": "12 fl oz"}
    """
    prefixes_to_update = set(updates.keys())
    kept = [t for t in current_tags
            if not any(t.startswith(f"{p}:") for p in prefixes_to_update)]
    new = [f"{k}:{v}" for k, v in updates.items() if v]
    return kept + new


def _row_to_product(row):
    """
    Maps a Supabase Product table row to the SmartInventory
    product dict format used throughout the app.
    """
    images = row.get("images") or []
    image_url = images[0] if images else None

    return {
        "id":           row["id"],
        "barcode":      row["sku"],
        "product_name": row["title"],
        "description":  row["description"],
        "product_type": row.get("categoryId"),
        "brand":        _extract_tag(row.get("tags"), "brand"),
        "size":         _extract_tag(row.get("tags"), "size"),
        "ocr_text":     _extract_tag(row.get("tags"), "ocr"),
        "image_url":    image_url,
        "timestamp":    row.get("createdAt"),
    }


# ── CRUD Operations ───────────────────────────────────────────────────────────


def find_product_by_barcode(barcode_value):
    """
    Looks up a product by SKU (barcode value).
    Returns the product as a dict or None if not found.
    """
    try:
        response = supabase.table("Product")\
            .select("*")\
            .eq("sku", barcode_value)\
            .execute()

        if not response.data:
            return None

        return _row_to_product(response.data[0])

    except Exception as e:
        print(f"Supabase error in find_product_by_barcode: {e}")
        return None


def insert_product(product_data):
    """
    Inserts a new product into the Product table.
    Maps SmartInventory fields to Supabase table columns.
    Stores brand, size, and ocr_text in the tags array.
    Stores image_url in the images array.
    Returns the new product ID or None on failure.
    """
    try:
        barcode      = product_data.get("barcode", "")
        product_name = product_data.get("product_name") or "Unknown Product"
        description  = product_data.get("description") or ""
        brand        = product_data.get("brand") or ""
        size         = product_data.get("size") or ""
        ocr_text     = product_data.get("ocr_text") or ""
        image_url    = product_data.get("image_url")

        # Build tags array to store extra SmartInventory fields
        tags = []
        if brand:
            tags.append(f"brand:{brand}")
        if size:
            tags.append(f"size:{size}")
        if ocr_text:
            tags.append(f"ocr:{ocr_text[:100]}")

        # Build images array
        images = []
        if image_url:
            images.append(image_url)

        row = {
            "id":          str(uuid.uuid4()),
            "sku":         barcode,
            "title":       product_name,
            "slug":        make_slug(product_name, barcode),
            "description": description,
            "price":       0,
            "categoryId":  DEFAULT_CATEGORY_ID,
            "tags":        tags,
            "images":      images,
            "isPublished": False,
        }

        response = supabase.table("Product").insert(row).execute()

        if response.data:
            print(f"Product saved to Supabase: {product_name} ({barcode})")
            return response.data[0]["id"]

        return None

    except Exception as e:
        error_str = str(e)
        # Handle duplicate barcode gracefully
        if "duplicate key" in error_str or "23505" in error_str:
            print(f"Product with barcode {product_data.get('barcode')} already exists.")
            existing = find_product_by_barcode(product_data.get("barcode", ""))
            if existing:
                return existing.get("id")
        print(f"Supabase error in insert_product: {e}")
        return None


def update_product(barcode_value, updated_fields):
    """
    Updates an existing product by SKU.
    Maps SmartInventory field names to Supabase table columns.
    """
    try:
        mapped = {}

        if "product_name" in updated_fields:
            name = updated_fields["product_name"]
            mapped["title"] = name
            mapped["slug"]  = make_slug(name, barcode_value)

        if "description" in updated_fields:
            mapped["description"] = updated_fields["description"]

        if "product_type" in updated_fields:
            mapped["categoryId"] = updated_fields["product_type"]

        if "image_url" in updated_fields:
            image_url = updated_fields["image_url"]
            mapped["images"] = [image_url] if image_url else []

        # Update brand and size inside tags array
        tag_updates = {}
        if "brand" in updated_fields:
            tag_updates["brand"] = updated_fields["brand"]
        if "size" in updated_fields:
            tag_updates["size"] = updated_fields["size"]
        if "ocr_text" in updated_fields:
            tag_updates["ocr"] = updated_fields["ocr_text"][:100]

        if tag_updates:
            current_tags = _get_raw_tags(barcode_value)
            mapped["tags"] = _update_tags(current_tags, tag_updates)

        if mapped:
            supabase.table("Product")\
                .update(mapped)\
                .eq("sku", barcode_value)\
                .execute()
            print(f"Product {barcode_value} updated.")

    except Exception as e:
        print(f"Supabase error in update_product: {e}")


def get_all_products():
    """
    Returns all products ordered by creation date descending.
    Maps Supabase table rows to SmartInventory product dicts.
    """
    try:
        response = supabase.table("Product")\
            .select("*")\
            .order("createdAt", desc=True)\
            .execute()

        if not response.data:
            return []

        return [_row_to_product(row) for row in response.data]

    except Exception as e:
        print(f"Supabase error in get_all_products: {e}")
        return []


def delete_product(barcode_value):
    """Deletes a product by SKU."""
    try:
        supabase.table("Product")\
            .delete()\
            .eq("sku", barcode_value)\
            .execute()
        print(f"Product {barcode_value} deleted.")

    except Exception as e:
        print(f"Supabase error in delete_product: {e}")