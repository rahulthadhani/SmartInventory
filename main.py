from camera.capture import start_camera, release_camera
from preprocessing.preprocess import preprocess_for_barcode, preprocess_for_ocr
from barcode.scanner import scan_barcode, validate_barcode, draw_barcode_overlay
from database.db import initialize_database
from database.queries import find_product_by_barcode, update_product, insert_product
from ocr.extractor import extract_text, extract_product_attributes
from llm.generator import generate_description
from config import MIN_OCR_LENGTH, MAX_OCR_RETRIES
import cv2


def display_product(product):
    """Prints a found product record to the terminal."""
    print("\n" + "=" * 45)
    print("  PRODUCT FOUND IN DATABASE")
    print("=" * 45)
    print(f"  Barcode      : {product['barcode']}")
    print(f"  Brand        : {product['brand'] or 'N/A'}")
    print(f"  Product Name : {product['product_name'] or 'N/A'}")
    print(f"  Product Type : {product['product_type'] or 'N/A'}")
    print(f"  Size         : {product['size'] or 'N/A'}")
    print(f"  Description  : {product['description'] or 'N/A'}")
    print(f"  Saved On     : {product['timestamp']}")
    print("=" * 45)


def prompt_update(product):
    """Asks the user if they want to update an existing product record."""
    while True:
        choice = (
            input("\nWould you like to update this product? (y/n): ").strip().lower()
        )
        if choice in ("y", "n"):
            break
        print("Please enter y or n.")

    if choice != "y":
        print("No changes made.")
    else:
        print("\nEnter new values (press Enter to keep current value):")
        updated_fields = {}

        for field in ["brand", "product_name", "product_type", "size"]:
            current = product.get(field) or ""
            new_value = input(f"  {field} [{current}]: ").strip()
            if new_value:
                updated_fields[field] = new_value

        if updated_fields:
            update_product(product["barcode"], updated_fields)
            print("Product updated successfully.")
        else:
            print("No changes entered.")

    # Refocus the OpenCV window after terminal input
    print("\nClick on the camera window and press 'S' to scan or 'Q' to quit.")
    cv2.waitKey(1)


def prompt_manual_entry(barcode):
    """
    Asks the user to manually enter product fields when
    OCR fails to extract enough useful information.
    Returns a product data dict.
    """
    print("\nPlease enter product details manually (press Enter to leave blank):")
    brand = input("  Brand: ").strip()
    product_name = input("  Product Name: ").strip()
    product_type = input("  Product Type: ").strip()
    size = input("  Size: ").strip()

    return {
        "barcode": barcode,
        "brand": brand or None,
        "product_name": product_name or None,
        "product_type": product_type or None,
        "size": size or None,
        "ocr_text": None,
        "description": None,
    }


def display_llm_result(barcode, llm_result, ocr_text):
    """
    Shows the user the OCR text and LLM-generated description
    side by side so they can evaluate the quality before saving.
    """
    print("\n" + "=" * 45)
    print("  OCR + LLM RESULT")
    print("=" * 45)
    print(f"  Barcode      : {barcode}")
    print(f"  Brand        : {llm_result['brand']}")
    print(f"  Product Name : {llm_result['product_name']}")
    print(f"  Product Type : {llm_result['product_type']}")
    print(f"  Size         : {llm_result['size']}")
    print(f"\n  OCR Text     : {ocr_text[:120]}{'...' if len(ocr_text) > 120 else ''}")
    print(f"\n  Description  : {llm_result['description']}")
    print("=" * 45)


def run_ocr_pipeline(cap, barcode_value):
    """
    Runs the full OCR + LLM pipeline for a new product.

    Flow:
    1. Ask user to show front of product and press S to scan
    2. Run OCR on the captured frame
    3. If OCR gets enough text → send to LLM → show result → confirm save
    4. If OCR fails after MAX_OCR_RETRIES → fall back to manual entry
    5. Save the final product record to the database
    """
    print("\nProduct not found in database.")
    print(">> Starting OCR pipeline...")
    print("Show the FRONT of the product to the camera, then press 'S' to scan.")

    ocr_text = ""
    retries = 0

    while retries <= MAX_OCR_RETRIES:
        # Wait for user to press S for OCR scan
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("SmartInventory - Show product front, press S", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                print("\nRunning OCR on captured frame...")
                ocr_frame = preprocess_for_ocr(frame)
                ocr_text = extract_text(ocr_frame)
                break
            elif key == ord("q"):
                return

        print(f"OCR extracted: '{ocr_text[:80]}{'...' if len(ocr_text) > 80 else ''}'")

        # Check if we got enough text to be useful
        if len(ocr_text) >= MIN_OCR_LENGTH:
            break

        retries += 1
        if retries <= MAX_OCR_RETRIES:
            print(f"\nNot enough text detected (attempt {retries}/{MAX_OCR_RETRIES}).")
            print("Try showing a different side of the packaging and press 'S' again.")
        else:
            print("\nOCR could not extract enough information after multiple attempts.")

    # Decide what to do based on OCR results
    if len(ocr_text) >= MIN_OCR_LENGTH:
        # We have enough text — send to LLM
        print("\nSending extracted text to LLM for description generation...")
        attributes = extract_product_attributes(ocr_text)
        llm_result = generate_description(barcode_value, attributes)

        if llm_result:
            while True:
                choice = (
                    input("\nSave this product to the database? (y/n): ")
                    .strip()
                    .lower()
                )
                if choice in ("y", "n"):
                    break
                print("Please enter y or n.")

            if choice == "y":
                product_data = {
                    "barcode": barcode_value,
                    "brand": llm_result["brand"],
                    "product_name": llm_result["product_name"],
                    "product_type": llm_result["product_type"],
                    "size": llm_result["size"],
                    "ocr_text": ocr_text,
                    "description": llm_result["description"],
                }
                insert_product(product_data)
                print("Product saved successfully.")
            else:
                print("Product not saved.")
        else:
            print("LLM generation failed. Falling back to manual entry.")
            product_data = prompt_manual_entry(barcode_value)
            _confirm_and_save(product_data)

    else:
        # OCR failed entirely — manual entry
        print("Switching to manual entry.")
        product_data = prompt_manual_entry(barcode_value)
        _confirm_and_save(product_data)

    # Refocus the OpenCV window after terminal input
    print("\nClick on the camera window and press 'S' to scan or 'Q' to quit.")
    cv2.waitKey(1)  # ← last line of this function


def _confirm_and_save(product_data):
    """
    Shows the manually entered data and asks for confirmation before saving.
    """
    print("\n" + "=" * 45)
    print("  MANUAL ENTRY SUMMARY")
    print("=" * 45)
    for key, value in product_data.items():
        if key not in ("ocr_text", "description"):
            print(f"  {key:<14}: {value or 'N/A'}")
    print("=" * 45)

    while True:
        choice = input("Save this product? (y/n): ").strip().lower()
        if choice in ("y", "n"):
            break
        print("Please enter y or n.")
    if choice == "y":
        insert_product(product_data)
        print("Product saved successfully.")
    else:
        print("Product not saved.")


def handle_barcode(barcode_value, cap):
    """
    Core decision logic for a decoded barcode:
    - Found in DB  → display it, offer update
    - Not found    → run OCR + LLM pipeline
    """
    barcode_value = barcode_value.strip()

    print(f"\nBarcode detected: {barcode_value}")

    is_valid, reason = validate_barcode(barcode_value)
    print(f"Validation: {reason}")

    if not is_valid:
        print("Skipping invalid barcode.")
        return

    print(f"Looking up barcode: '{barcode_value}' (length: {len(barcode_value)})")
    product = find_product_by_barcode(barcode_value)

    if product:
        display_product(product)
        prompt_update(product)
    else:
        run_ocr_pipeline(cap, barcode_value)


def main():
    print("=== SmartInventory - Week 3 ===")

    # Initialize database
    initialize_database()

    # Open camera
    cap = start_camera(camera_index=0)
    if cap is None:
        return

    print("\nLive preview started.")
    print("Press 'S' to scan a barcode | Press 'Q' to quit.")

    scanned = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("SmartInventory - Scan a Barcode", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("\nScanning frame for barcode...")

            # Try raw frame first, then preprocessed
            barcode_results = scan_barcode(frame)
            if not barcode_results:
                processed = preprocess_for_barcode(frame)
                barcode_results = scan_barcode(processed)

            if barcode_results:
                annotated = draw_barcode_overlay(frame.copy(), barcode_results)
                cv2.imshow("SmartInventory - Scan a Barcode", annotated)
                cv2.waitKey(1000)

                for result in barcode_results:
                    handle_barcode(result["value"], cap)

                scanned = True
            else:
                print("No barcode detected. Try adjusting the angle or distance.")

        elif key == ord("q"):
            print("Exiting.")
            break

    release_camera(cap)

    if not scanned:
        print("\nNo barcodes were scanned this session.")


if __name__ == "__main__":
    main()
