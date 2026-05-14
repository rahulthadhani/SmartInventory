import uuid
import re
from database.supabase_client import supabase
from config import DEFAULT_CATEGORY_ID


def make_slug(title, sku):
    """
    Generates a URL-friendly slug from the product title.
    Falls back to the SKU if title is missing.
    Example: 'Celsius Sparkling Orange' → 'celsius-sparkling-orange'
    """
    base = title or sku
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    # Append part of SKU to ensure uniqueness
    return f"{slug}-{sku[-4:]}"


def find_product_by_barcode(barcode_value):
    """
    Looks up a product by SKU (barcode value).
    Returns the product as a dict or None if not found.
    Maps your table fields back to SmartInventory field names.
    """
    response = supabase.table("Product").select("*").eq("sku", barcode_value).execute()

    if not response.data:
        return None

    row = response.data[0]

    # Map your table fields to SmartInventory field names
    return {
        "id": row["id"],
        "barcode": row["sku"],
        "product_name": row["title"],
        "description": row["description"],
        "product_type": row.get("categoryId"),
        "brand": _extract_tag(row.get("tags"), "brand"),
        "size": _extract_tag(row.get("tags"), "size"),
        "ocr_text": _extract_tag(row.get("tags"), "ocr"),
        "timestamp": row.get("createdAt"),
    }


def insert_product(product_data):
    """
    Inserts a new product into your Product table.
    Maps SmartInventory fields to your table's required fields.
    """
    barcode = product_data.get("barcode", "")
    product_name = product_data.get("product_name") or "Unknown Product"
    description = product_data.get("description") or ""
    brand = product_data.get("brand") or ""
    size = product_data.get("size") or ""
    ocr_text = product_data.get("ocr_text") or ""

    # Build tags array to store extra SmartInventory fields
    tags = []
    if brand:
        tags.append(f"brand:{brand}")
    if size:
        tags.append(f"size:{size}")
    if ocr_text:
        tags.append(f"ocr:{ocr_text[:100]}")

    row = {
        "id": str(uuid.uuid4()),
        "sku": barcode,
        "title": product_name,
        "slug": make_slug(product_name, barcode),
        "description": description,
        "price": 0,
        "categoryId": DEFAULT_CATEGORY_ID,
        "tags": tags,
        "isPublished": False,
    }

    response = supabase.table("Product").insert(row).execute()

    if response.data:
        print(f"Product saved to Supabase with id {response.data[0]['id']}")
        return response.data[0]["id"]
    return None


def update_product(barcode_value, updated_fields):
    """
    Updates an existing product by SKU.
    Maps SmartInventory field names to your table fields.
    """
    mapped = {}

    if "product_name" in updated_fields:
        mapped["title"] = updated_fields["product_name"]
        mapped["slug"] = make_slug(updated_fields["product_name"], barcode_value)

    if "description" in updated_fields:
        mapped["description"] = updated_fields["description"]

    if "product_type" in updated_fields:
        mapped["categoryId"] = updated_fields["product_type"]

    # Update brand and size inside tags array
    tag_updates = {}
    if "brand" in updated_fields:
        tag_updates["brand"] = updated_fields["brand"]
    if "size" in updated_fields:
        tag_updates["size"] = updated_fields["size"]

    if tag_updates:
        existing = find_product_by_barcode(barcode_value)
        if existing:
            current_tags = _get_raw_tags(barcode_value)
            new_tags = _update_tags(current_tags, tag_updates)
            mapped["tags"] = new_tags

    if mapped:
        supabase.table("Product").update(mapped).eq("sku", barcode_value).execute()
        print(f"Product {barcode_value} updated.")


def get_all_products():
    """
    Returns all products ordered by creation date descending.
    Maps table fields back to SmartInventory field names.
    """
    response = (
        supabase.table("Product").select("*").order("createdAt", desc=True).execute()
    )

    if not response.data:
        return []

    return [
        {
            "id": row["id"],
            "barcode": row["sku"],
            "product_name": row["title"],
            "description": row["description"],
            "product_type": row.get("categoryId"),
            "brand": _extract_tag(row.get("tags"), "brand"),
            "size": _extract_tag(row.get("tags"), "size"),
            "ocr_text": _extract_tag(row.get("tags"), "ocr"),
            "timestamp": row.get("createdAt"),
        }
        for row in response.data
    ]


def delete_product(barcode_value):
    """Deletes a product by SKU."""
    supabase.table("Product").delete().eq("sku", barcode_value).execute()
    print(f"Product {barcode_value} deleted.")


# ── Tag helpers ───────────────────────────────────────────────────────────────


def _extract_tag(tags, prefix):
    """
    Extracts a value from the tags array by prefix.
    Example: tags = ["brand:Celsius", "size:12 fl oz"]
             _extract_tag(tags, "brand") → "Celsius"
    """
    if not tags:
        return None
    for tag in tags:
        if tag.startswith(f"{prefix}:"):
            return tag[len(prefix) + 1 :]
    return None


def _get_raw_tags(barcode_value):
    """Returns the raw tags array for a product."""
    response = (
        supabase.table("Product").select("tags").eq("sku", barcode_value).execute()
    )
    if response.data:
        return response.data[0].get("tags") or []
    return []


def _update_tags(current_tags, updates):
    """
    Updates specific tag values while preserving others.
    updates = {"brand": "Celsius", "size": "12 fl oz"}
    """
    prefixes_to_update = set(updates.keys())
    kept = [
        t
        for t in current_tags
        if not any(t.startswith(f"{p}:") for p in prefixes_to_update)
    ]
    new = [f"{k}:{v}" for k, v in updates.items() if v]
    return kept + new
