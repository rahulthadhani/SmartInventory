from camera.capture import start_camera, live_preview, release_camera
from preprocessing.preprocess import preprocess_for_barcode
from barcode.scanner import scan_barcode, validate_barcode, draw_barcode_overlay
from database.db import initialize_database
from database.queries import find_product_by_barcode, update_product
import cv2


def display_product(product):
    """
    Prints a found product record to the terminal in a readable format.
    """
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
    """
    Asks the user if they want to update the existing product record.
    If yes, prompts for new values for each field.
    Pressing Enter with no input keeps the current value.
    """
    print("\nWould you like to update this product? (y/n): ", end="")
    choice = input().strip().lower()

    if choice != "y":
        print("No changes made.")
        return

    print("\nEnter new values (press Enter to keep current value):")

    updated_fields = {}

    fields = ["brand", "product_name", "product_type", "size"]
    for field in fields:
        current = product.get(field) or ""
        new_value = input(f"  {field} [{current}]: ").strip()
        if new_value:
            updated_fields[field] = new_value

    if updated_fields:
        update_product(product["barcode"], updated_fields)
        print("Product updated successfully.")
    else:
        print("No changes entered.")


def handle_barcode(barcode_value):
    """
    Core decision logic for a decoded barcode:
    - If found in DB → display it, offer update
    - If not found   → flag for OCR pipeline (Week 3)
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
        print(f"\nBarcode {barcode_value} not found in database.")
        print(">> OCR pipeline will run here in Week 3.")


def main():
    print("=== SmartInventory - Week 2 ===")

    # Step 1: Initialize the database (creates file + table if not exists)
    initialize_database()

    # Step 2: Open camera
    cap = start_camera(camera_index=0)
    if cap is None:
        return

    # Step 3: Live preview — press S to scan, Q to quit
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

            # Try raw frame first — preprocessing can sometimes hurt detection
            barcode_results = scan_barcode(frame)
            # If raw finds nothing, try the preprocessed version
            if not barcode_results:
                processed = preprocess_for_barcode(frame)
                barcode_results = scan_barcode(processed)

            if barcode_results:
                # Draw overlay on the live frame for visual feedback
                annotated = draw_barcode_overlay(frame.copy(), barcode_results)
                cv2.imshow("SmartInventory - Scan a Barcode", annotated)
                cv2.waitKey(1000)  # show the overlay for 1 second

                for result in barcode_results:
                    handle_barcode(result["value"])

                scanned = True
            else:
                print("No barcode detected. Try adjusting the angle or distance.")

        elif key == ord("q"):
            print("Exiting.")
            break

    # Step 4: Release camera
    release_camera(cap)

    if not scanned:
        print("\nNo barcodes were scanned this session.")


if __name__ == "__main__":
    main()
    