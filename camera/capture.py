import cv2
import os


def start_camera(camera_index=0):
    """
    Opens a connection to the device camera.
    camera_index=0 means the default/built-in camera.
    If you have multiple cameras, try index 1 or 2.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print("Camera opened successfully.")
    return cap


def capture_frame(cap):
    """
    Reads a single frame from the camera feed.
    Returns the frame as a NumPy array (height x width x 3 color channels).
    ret is a boolean — True if the frame was read successfully.
    """
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        return None

    return frame


def release_camera(cap):
    """
    Always release the camera when done.
    This frees the hardware so other apps can use it.
    """
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")


def save_frame(frame, filename, output_dir="data/sample_frames"):
    """
    Saves a captured frame as a .jpg file to disk.
    Used for testing preprocessing and barcode detection later.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Frame saved to {filepath}")
    return filepath


def live_preview(cap, window_name="SmartInventory - Camera Feed"):
    """
    Opens a live window showing the camera feed.
    Press 'S' to save the current frame.
    Press 'Q' to quit.
    Returns the last saved frame.
    """
    saved_frame = None
    frame_count = 0

    print("Live preview started. Press 'S' to save a frame, 'Q' to quit.")

    while True:
        frame = capture_frame(cap)

        if frame is None:
            break

        # Show the live feed in a window
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = f"frame_{frame_count:04d}.jpg"
            save_frame(frame, filename)
            saved_frame = frame
            frame_count += 1
            print(f"Saved frame {frame_count}")

        elif key == ord("q"):
            print("Exiting preview.")
            break

    return saved_frame
