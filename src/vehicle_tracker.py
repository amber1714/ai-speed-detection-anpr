# NEXT STEP: VEHICLE TRACKING WITH UNIQUE IDs
#
# Save this file as:
# src/vehicle_tracker.py
#
# Run it from the VS Code terminal using:
# python src/vehicle_tracker.py

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_ROOT / "data" / "input" / "traffic.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "vehicle_tracking.mp4"

MODEL_NAME = "yolo26n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE_THRESHOLD = 0.35

MAX_DISTANCE = 100
MAX_MISSING_FRAMES = 20


class VehicleTracker:
    def __init__(self):
        self.next_id = 1
        self.vehicles = {}

    def update(self, detections):
        updated_vehicles = {}
        assigned_ids = set()

        for detection in detections:
            x1, y1, x2, y2, class_name, confidence = detection

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_center = np.array([center_x, center_y])

            best_id = None
            best_distance = float("inf")

            for vehicle_id, data in self.vehicles.items():

                if vehicle_id in assigned_ids:
                    continue

                previous_center = np.array(data["center"])

                distance = np.linalg.norm(
                    current_center - previous_center
                )

                if distance < best_distance and distance < MAX_DISTANCE:
                    best_distance = distance
                    best_id = vehicle_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            updated_vehicles[best_id] = {
                "center": (center_x, center_y),
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "confidence": confidence,
                "missing": 0,
            }

            assigned_ids.add(best_id)

        for vehicle_id, data in self.vehicles.items():

            if vehicle_id not in updated_vehicles:

                missing_count = data["missing"] + 1

                if missing_count <= MAX_MISSING_FRAMES:
                    data["missing"] = missing_count
                    updated_vehicles[vehicle_id] = data

        self.vehicles = updated_vehicles

        return self.vehicles


def track_vehicles():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Traffic video not found:\n{INPUT_VIDEO}"
        )

    print("Loading YOLO model...")

    model = YOLO(MODEL_NAME)

    tracker = VehicleTracker()

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Unable to open traffic video.")

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

    print("Vehicle tracking started...")

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

        detections = []

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                detections.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        class_name,
                        confidence,
                    )
                )

        tracked_vehicles = tracker.update(detections)

        for vehicle_id, vehicle in tracked_vehicles.items():

            if vehicle["missing"] > 0:
                continue

            x1, y1, x2, y2 = vehicle["bbox"]

            center_x, center_y = vehicle["center"]

            class_name = vehicle["class_name"]

            confidence = vehicle["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"ID {vehicle_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1,
            )

        writer.write(frame)

        cv2.imshow(
            "AI Vehicle Tracking",
            frame,
        )

        if total_frames > 0:

            progress = (
                frame_number / total_frames
            ) * 100

            print(
                f"\rProcessing "
                f"{frame_number}/{total_frames} "
                f"({progress:.1f}%)",
                end="",
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\nVehicle tracking completed.")

    print(
        f"Output saved to:\n"
        f"{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    track_vehicles()