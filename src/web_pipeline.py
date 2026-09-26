# Save this file as:
# src/web_pipeline.py
#
# This is the deployment-friendly version of the pipeline.
# It does NOT use cv2.imshow(), so it can run on Streamlit/Railway.

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.plate_reader import detect_and_read_plate


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
VIOLATION_DIR = OUTPUT_DIR / "violations"

MODEL_NAME = "yolo26n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE_THRESHOLD = 0.35

DEFAULT_DISTANCE_METERS = 10.0
DEFAULT_SPEED_LIMIT_KMPH = 50.0


@lru_cache(maxsize=1)
def load_yolo_model():
    return YOLO(MODEL_NAME)


def crossed_line(previous_y, current_y, line_y):
    return (
        previous_y < line_y <= current_y
        or previous_y > line_y >= current_y
    )


def calculate_speed(distance_meters, elapsed_seconds):
    if elapsed_seconds <= 0:
        return 0.0

    speed_mps = distance_meters / elapsed_seconds

    return speed_mps * 3.6


def crop_vehicle(
    frame,
    x1,
    y1,
    x2,
    y2,
):
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[
        y1:y2,
        x1:x2,
    ]


def save_evidence(
    frame,
    vehicle_id,
    plate,
    speed,
):
    VIOLATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_plate = (
        plate
        if plate
        else "UNKNOWN"
    )

    filename = (
        f"vehicle_{vehicle_id}_"
        f"{safe_plate}_"
        f"{speed:.1f}kmph_"
        f"{timestamp}.jpg"
    )

    path = (
        VIOLATION_DIR
        / filename
    )

    cv2.imwrite(
        str(path),
        frame,
    )

    return str(path)


def process_video(
    input_video,
    output_video,
    calibrated_distance_meters=DEFAULT_DISTANCE_METERS,
    speed_limit_kmph=DEFAULT_SPEED_LIMIT_KMPH,
    progress_callback=None,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIOLATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_video = Path(
        input_video
    )

    output_video = Path(
        output_video
    )

    if not input_video.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    model = load_yolo_model()

    cap = cv2.VideoCapture(
        str(input_video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open uploaded video."
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    line_a = int(
        height * 0.40
    )

    line_b = int(
        height * 0.70
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Unable to create output video."
        )

    previous_positions = {}

    first_crossings = {}

    vehicle_speeds = {}

    vehicle_plates = {}

    ocr_attempts = {}

    logged_vehicles = set()

    records = []

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        current_time = (
            frame_number
            / fps
        )

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        cv2.line(
            frame,
            (
                0,
                line_a,
            ),
            (
                width,
                line_a,
            ),
            (
                255,
                255,
                0,
            ),
            3,
        )

        cv2.line(
            frame,
            (
                0,
                line_b,
            ),
            (
                width,
                line_b,
            ),
            (
                255,
                0,
                255,
            ),
            3,
        )

        cv2.putText(
            frame,
            "SPEED LINE A",
            (
                20,
                max(
                    25,
                    line_a - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                0,
            ),
            2,
        )

        cv2.putText(
            frame,
            "SPEED LINE B",
            (
                20,
                max(
                    25,
                    line_b - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                0,
                255,
            ),
            2,
        )

        boxes = results[0].boxes

        if (
            boxes is not None
            and boxes.id is not None
        ):
            xyxy_boxes = (
                boxes.xyxy
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .cpu()
                .tolist()
            )

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            for (
                box,
                class_id,
                vehicle_id,
            ) in zip(
                xyxy_boxes,
                class_ids,
                track_ids,
            ):
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = map(
                    int,
                    box,
                )

                class_id = int(
                    class_id
                )

                class_name = (
                    model.names[
                        class_id
                    ]
                )

                center_x = int(
                    (
                        x1
                        + x2
                    )
                    / 2
                )

                bottom_y = y2

                if (
                    vehicle_id
                    in previous_positions
                ):
                    old_y = (
                        previous_positions[
                            vehicle_id
                        ]
                    )

                    crossed_a = crossed_line(
                        old_y,
                        bottom_y,
                        line_a,
                    )

                    crossed_b = crossed_line(
                        old_y,
                        bottom_y,
                        line_b,
                    )

                    if (
                        vehicle_id
                        not in vehicle_speeds
                    ):
                        if (
                            vehicle_id
                            not in first_crossings
                        ):
                            if crossed_a:
                                first_crossings[
                                    vehicle_id
                                ] = {
                                    "line": "A",
                                    "time": current_time,
                                }

                            elif crossed_b:
                                first_crossings[
                                    vehicle_id
                                ] = {
                                    "line": "B",
                                    "time": current_time,
                                }

                        else:
                            first_line = (
                                first_crossings[
                                    vehicle_id
                                ][
                                    "line"
                                ]
                            )

                            first_time = (
                                first_crossings[
                                    vehicle_id
                                ][
                                    "time"
                                ]
                            )

                            measurement_complete = False

                            if (
                                first_line == "A"
                                and crossed_b
                            ):
                                measurement_complete = True

                            elif (
                                first_line == "B"
                                and crossed_a
                            ):
                                measurement_complete = True

                            if measurement_complete:
                                elapsed_time = (
                                    current_time
                                    - first_time
                                )

                                speed = calculate_speed(
                                    calibrated_distance_meters,
                                    elapsed_time,
                                )

                                vehicle_speeds[
                                    vehicle_id
                                ] = speed

                previous_positions[
                    vehicle_id
                ] = bottom_y

                if (
                    vehicle_id
                    in vehicle_speeds
                    and vehicle_id
                    not in vehicle_plates
                ):
                    attempts = (
                        ocr_attempts.get(
                            vehicle_id,
                            0,
                        )
                    )

                    if attempts < 5:
                        vehicle_crop = crop_vehicle(
                            frame,
                            x1,
                            y1,
                            x2,
                            y2,
                        )

                        if (
                            vehicle_crop
                            is not None
                            and vehicle_crop.size > 0
                        ):
                            (
                                plate_text,
                                plate_confidence,
                                _,
                            ) = detect_and_read_plate(
                                vehicle_crop
                            )

                            ocr_attempts[
                                vehicle_id
                            ] = (
                                attempts
                                + 1
                            )

                            if plate_text:
                                vehicle_plates[
                                    vehicle_id
                                ] = plate_text

                box_color = (
                    0,
                    255,
                    0,
                )

                label = (
                    f"ID {vehicle_id}"
                    f" | {class_name}"
                )

                if (
                    vehicle_id
                    in vehicle_speeds
                ):
                    speed = (
                        vehicle_speeds[
                            vehicle_id
                        ]
                    )

                    label += (
                        f" | "
                        f"{speed:.1f} km/h"
                    )

                    if (
                        speed
                        > speed_limit_kmph
                    ):
                        box_color = (
                            0,
                            0,
                            255,
                        )

                        label += (
                            " | OVERSPEED"
                        )

                if (
                    vehicle_id
                    in vehicle_plates
                ):
                    label += (
                        f" | "
                        f"{vehicle_plates[vehicle_id]}"
                    )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    box_color,
                    2,
                )

                cv2.circle(
                    frame,
                    (
                        center_x,
                        bottom_y,
                    ),
                    5,
                    box_color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    box_color,
                    2,
                )

                if (
                    vehicle_id
                    in vehicle_speeds
                    and vehicle_id
                    not in logged_vehicles
                ):
                    attempts = (
                        ocr_attempts.get(
                            vehicle_id,
                            0,
                        )
                    )

                    ready_to_log = (
                        vehicle_id
                        in vehicle_plates
                        or attempts >= 5
                    )

                    if ready_to_log:
                        speed = (
                            vehicle_speeds[
                                vehicle_id
                            ]
                        )

                        plate = (
                            vehicle_plates.get(
                                vehicle_id,
                                "UNKNOWN",
                            )
                        )

                        violation_status = (
                            "OVERSPEED"
                            if speed
                            > speed_limit_kmph
                            else "NORMAL"
                        )

                        evidence_image = ""

                        if (
                            violation_status
                            == "OVERSPEED"
                        ):
                            evidence_image = (
                                save_evidence(
                                    frame.copy(),
                                    vehicle_id,
                                    plate,
                                    speed,
                                )
                            )

                        record = {
                            "vehicle_id": vehicle_id,
                            "number_plate": plate,
                            "speed_kmph": round(
                                speed,
                                2,
                            ),
                            "speed_limit_kmph": round(
                                speed_limit_kmph,
                                2,
                            ),
                            "timestamp": (
                                datetime.now()
                                .strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            ),
                            "violation_status": (
                                violation_status
                            ),
                            "evidence_image": (
                                evidence_image
                            ),
                        }

                        records.append(
                            record
                        )

                        logged_vehicles.add(
                            vehicle_id
                        )

        measured_count = len(
            vehicle_speeds
        )

        overspeed_count = sum(
            1
            for speed
            in vehicle_speeds.values()
            if speed
            > speed_limit_kmph
        )

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                430,
                115,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.putText(
            frame,
            "AI SPEED DETECTION + ANPR",
            (
                20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Speed Limit: "
                f"{speed_limit_kmph:.0f} km/h"
            ),
            (
                20,
                65,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Measured: "
                f"{measured_count}"
                f" | Overspeed: "
                f"{overspeed_count}"
            ),
            (
                20,
                92,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255,
            ),
            2,
        )

        writer.write(
            frame
        )

        if (
            progress_callback is not None
            and total_frames > 0
        ):
            progress = min(
                frame_number
                / total_frames,
                1.0,
            )

            progress_callback(
                progress
            )

    cap.release()

    writer.release()

    dataframe = pd.DataFrame(
        records
    )

    csv_path = (
        OUTPUT_DIR
        / "violations.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "total_records": len(
            dataframe
        ),
        "measured_vehicles": len(
            vehicle_speeds
        ),
        "plates_detected": len(
            vehicle_plates
        ),
        "overspeed_violations": sum(
            1
            for speed
            in vehicle_speeds.values()
            if speed
            > speed_limit_kmph
        ),
        "output_video": str(
            output_video
        ),
        "csv_file": str(
            csv_path
        ),
    }

    if progress_callback is not None:
        progress_callback(
            1.0
        )

    return (
        dataframe,
        summary,
    )