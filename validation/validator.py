def validate_barcode_format(barcode_value):
    """
    Checks that the barcode is a known valid format.
    Returns (is_valid, message).
    """
    barcode_value = barcode_value.strip()

    if not barcode_value:
        return False, "Barcode is empty."

    if not barcode_value.isdigit():
        if len(barcode_value) >= 6:
            return True, "Non-numeric barcode accepted."
        return False, "Barcode contains invalid characters."

    length = len(barcode_value)
    valid_lengths = {8: "EAN-8", 12: "UPC-A", 13: "EAN-13"}

    if length in valid_lengths:
        return True, f"Valid {valid_lengths[length]} barcode."

    return False, f"Unrecognized barcode length: {length} digits."


def validate_required_fields(product_data):
    """
    Checks that brand and product_name are present.
    Returns (is_valid, list of missing fields).
    """
    required = ["brand", "product_name"]
    missing = [
        field
        for field in required
        if not product_data.get(field)
        or product_data[field].strip().lower() == "unknown"
    ]

    if missing:
        return False, missing

    return True, []


def validate_no_duplicate(barcode_value, find_product_fn):
    """
    Checks the database to see if a barcode already exists.
    Returns (is_duplicate, existing_product or None).
    """
    existing = find_product_fn(barcode_value)
    if existing:
        return True, existing
    return False, None


def validate_description_quality(description):
    """
    Checks that the LLM-generated description meets
    minimum quality standards.

    Rules:
    - Must be at least 20 characters
    - Must not be "Unknown"
    - Must contain at least 5 words
    """
    if not description:
        return False, "Description is empty."

    if description.strip().lower() == "unknown":
        return False, "Description was not generated."

    if len(description.strip()) < 20:
        return False, "Description is too short."

    if len(description.split()) < 5:
        return False, "Description does not contain enough words."

    return True, "Description quality check passed."


def run_all_validations(product_data, find_product_fn):
    """
    Runs all four validation checks on a product record.
    Returns a dict with results for each check and an
    overall is_valid boolean.
    """
    results = {}

    # 1. Barcode format
    barcode_valid, barcode_msg = validate_barcode_format(
        product_data.get("barcode", "")
    )
    results["barcode_format"] = {"passed": barcode_valid, "message": barcode_msg}

    # 2. Required fields
    fields_valid, missing_fields = validate_required_fields(product_data)
    results["required_fields"] = {
        "passed": fields_valid,
        "message": (
            f"Missing: {', '.join(missing_fields)}"
            if missing_fields
            else "All required fields present."
        ),
    }

    # 3. Duplicate check
    is_duplicate, existing = validate_no_duplicate(
        product_data.get("barcode", ""), find_product_fn
    )
    results["duplicate_check"] = {
        "passed": not is_duplicate,
        "message": (
            "Duplicate found in database." if is_duplicate else "No duplicate found."
        ),
    }

    # 4. Description quality
    desc_valid, desc_msg = validate_description_quality(
        product_data.get("description", "")
    )
    results["description_quality"] = {"passed": desc_valid, "message": desc_msg}

    # Overall result — all checks must pass
    results["is_valid"] = all(
        v["passed"] for v in results.values() if isinstance(v, dict)
    )

    return results
