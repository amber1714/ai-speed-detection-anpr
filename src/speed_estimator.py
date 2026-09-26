# src/speed_estimator.py
#
# Save this file as:
# src/speed_estimator.py
#
# Run:
# python src/speed_estimator.py
#
# IMPORTANT:
# CALIBRATED_DISTANCE_METERS must eventually be replaced with the
# real-world road distance between LINE_A and LINE_B.
#
# For now, we will use 10 metres for testing.

from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_ROOT / "data" / "input" / "traffic.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "speed_detection.mp4"

MODEL_NAME = "yolo26n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE_THRESHOLD = 0.35

# Temporary calibration distance.
# Later we will accurately calibrate this for the selected road/video.
CALIBRATED_DISTANCE_METERS = 10.0

# Example speed limit for testing
SPEED_LIMIT_KMPH = 50


def crossed_line(previous_y, current_y, line_y):
    return (
        previous_y < line_y <= current_y
        or previous_y > line_y >= current_y
    )


def calculate_speed(distance_meters, time_seconds):
    if time_seconds <= 0:
        return 0.0

    speed_mps = distance_meters / time_seconds
    speed_kmph = speed_mps * 3.6

    return speed_kmph


def run_speed_estimation():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Traffic video not found:\n{INPUT_VIDEO}"
        )

    print("Loading YOLO model...")

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Unable to open traffic video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Two virtual road lines
    LINE_A = int(height * 0.40)
    LINE_B = int(height * 0.70)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0

    previous_y = {}

    first_crossing = {}

    vehicle_speeds = {}

    print("Speed estimation started...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        current_time = frame_number / fps

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # Draw virtual speed measurement lines
        cv2.line(
            frame,
            (0, LINE_A),
            (width, LINE_A),
            (255, 255, 0),
            3,
        )

        cv2.line(
            frame,
            (0, LINE_B),
            (width, LINE_B),
            (255, 0, 255),
            3,
        )

        cv2.putText(
            frame,
            "LINE A",
            (20, LINE_A - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "LINE B",
            (20, LINE_B - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

        boxes = results[0].boxes

        if (
            boxes is not None
            and boxes.id is not None
        ):

            xyxy_boxes = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()

            for (
                box,
                class_id,
                confidence,
                track_id,
            ) in zip(
                xyxy_boxes,
                class_ids,
                confidences,
                track_ids,
            ):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(class_id)

                class_name = model.names[class_id]

                # Bottom-center point is more suitable for road-line crossing
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)

                if track_id in previous_y:

                    old_y = previous_y[track_id]

                    crossed_a = crossed_line(
                        old_y,
                        bottom_y,
                        LINE_A,
                    )

                    crossed_b = crossed_line(
                        old_y,
                        bottom_y,
                        LINE_B,
                    )

                    if track_id not in vehicle_speeds:

                        # First line crossing
                        if track_id not in first_crossing:

                            if crossed_a:
                                first_crossing[track_id] = {
                                    "line": "A",
                                    "time": current_time,
                                }

                            elif crossed_b:
                                first_crossing[track_id] = {
                                    "line": "B",
                                    "time": current_time,
                                }

                        else:

                            first_line = first_crossing[
                                track_id
                            ]["line"]

                            first_time = first_crossing[
                                track_id
                            ]["time"]

                            completed_measurement = False

                            if (
                                first_line == "A"
                                and crossed_b
                            ):
                                completed_measurement = True

                            elif (
                                first_line == "B"
                                and crossed_a
                            ):
                                completed_measurement = True

                            if completed_measurement:

                                elapsed_time = (
                                    current_time
                                    - first_time
                                )

                                speed = calculate_speed(
                                    CALIBRATED_DISTANCE_METERS,
                                    elapsed_time,
                                )

                                vehicle_speeds[
                                    track_id
                                ] = speed

                                print(
                                    f"\nVehicle ID "
                                    f"{track_id}: "
                                    f"{speed:.2f} km/h"
                                )

                previous_y[track_id] = bottom_y

                # Default box color
                box_color = (0, 255, 0)

                label = (
                    f"ID {track_id} | "
                    f"{class_name}"
                )

                if track_id in vehicle_speeds:

                    speed = vehicle_speeds[track_id]

                    label += f" | {speed:.1f} km/h"

                    if speed > SPEED_LIMIT_KMPH:

                        box_color = (0, 0, 255)

                        label += " | OVERSPEED"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                cv2.circle(
                    frame,
                    (center_x, bottom_y),
                    5,
                    box_color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

        cv2.putText(
            frame,
            f"Speed Limit: {SPEED_LIMIT_KMPH} km/h",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        cv2.imshow(
            "AI Speed Detection",
            frame,
        )

        if total_frames > 0:

            progress = (
                frame_number
                / total_frames
            ) * 100

            print(
                f"\rProcessing "
                f"{frame_number}/"
                f"{total_frames} "
                f"({progress:.1f}%)",
                end="",
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\n")
    print("Speed estimation completed.")

    print(
        f"Output saved to:\n"
        f"{OUTPUT_VIDEO}"
    )

    print("\nDetected vehicle speeds:")

    for vehicle_id, speed in vehicle_speeds.items():

        print(
            f"Vehicle ID {vehicle_id}: "
            f"{speed:.2f} km/h"
        )


if __name__ == "__main__":
    run_speed_estimation()