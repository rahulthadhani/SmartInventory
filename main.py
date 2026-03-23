from camera.capture import start_camera, live_preview, release_camera, save_frame
from preprocessing.preprocess import (
    preprocess_for_barcode,
    preprocess_for_ocr,
    save_preprocessed,
)


def main():
    print("=== SmartInventory - Week 1 ===")

    # Step 1: Open the camera
    cap = start_camera(camera_index=0)
    if cap is None:
        return

    # Step 2: Show live preview, let user capture a frame with 'S'
    raw_frame = live_preview(cap)

    # Step 3: Release camera
    release_camera(cap)

    # Step 4: If a frame was saved, run preprocessing on it
    if raw_frame is not None:
        print("\nRunning preprocessing pipeline...")

        # Save the raw original for comparison
        save_frame(raw_frame, "raw_original.jpg")

        # Preprocess for barcode detection
        barcode_ready = preprocess_for_barcode(raw_frame)
        save_preprocessed(barcode_ready, "preprocessed_barcode.jpg")

        # Preprocess for OCR
        ocr_ready = preprocess_for_ocr(raw_frame)
        save_preprocessed(ocr_ready, "preprocessed_ocr.jpg")

        print("\nWeek 1 complete. Check data/sample_frames/ for your output images.")
    else:
        print("No frame was captured. Run the program again and press 'S' to save.")


if __name__ == "__main__":
    main()
