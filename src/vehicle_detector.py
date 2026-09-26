# src/vehicle_detector.py

from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_ROOT / "data" / "input" / "traffic.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "vehicle_detection.mp4"

MODEL_NAME = "yolo26n.pt"

# COCO vehicle class IDs
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE_THRESHOLD = 0.35


def detect_vehicles():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"\nTraffic video not found.\n"
            f"Put your video here:\n{INPUT_VIDEO}\n"
            f"Rename it to: traffic.mp4"
        )

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open the traffic video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0

    print("Vehicle detection started...")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        cv2.imshow("AI Vehicle Detection", annotated_frame)

        if total_frames > 0:
            progress = (frame_number / total_frames) * 100
            print(
                f"\rProcessing frame {frame_number}/{total_frames} "
                f"({progress:.1f}%)",
                end="",
            )

        # Press Q to stop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n")
    print("Vehicle detection completed.")
    print(f"Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    detect_vehicles()