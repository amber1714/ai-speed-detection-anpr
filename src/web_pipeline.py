# src/web_pipeline.py

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


def crossed_line(
    previous_position,
    current_position,
    line_position,
):
    return (
        previous_position < line_position <= current_position
        or
        previous_position > line_position >= current_position
    )


def calculate_speed(
    distance_meters,
    elapsed_seconds,
):
    if elapsed_seconds <= 0:
        return 0.0

    return (
        distance_meters
        / elapsed_seconds
        * 3.6
    )


def process_crossing_pair(
    crossing_store,
    vehicle_id,
    crossed_a,
    crossed_b,
    current_time,
):
    if vehicle_id not in crossing_store:

        if crossed_a:
            crossing_store[vehicle_id] = {
                "line": "A",
                "time": current_time,
            }

        elif crossed_b:
            crossing_store[vehicle_id] = {
                "line": "B",
                "time": current_time,
            }

        return None

    first_crossing = crossing_store[vehicle_id]

    first_line = first_crossing["line"]
    first_time = first_crossing["time"]

    if current_time - first_time > 15:

        crossing_store.pop(
            vehicle_id,
            None,
        )

        return None

    completed = False

    if (
        first_line == "A"
        and crossed_b
    ):
        completed = True

    elif (
        first_line == "B"
        and crossed_a
    ):
        completed = True

    if not completed:
        return None

    return current_time - first_time


def crop_vehicle(
    frame,
    x1,
    y1,
    x2,
    y2,
):
    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(x1, width - 1),
    )

    x2 = max(
        0,
        min(x2, width),
    )

    y1 = max(
        0,
        min(y1, height - 1),
    )

    y2 = max(
        0,
        min(y2, height),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
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
    horizontal_line_a_pct=45.0,
    horizontal_line_b_pct=60.0,
    vertical_line_a_pct=40.0,
    vertical_line_b_pct=60.0,
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

    # =====================================================
    # USER-ADJUSTABLE SPEED LINES
    # =====================================================

    horizontal_line_a = int(
        height
        * horizontal_line_a_pct
        / 100
    )

    horizontal_line_b = int(
        height
        * horizontal_line_b_pct
        / 100
    )

    vertical_line_a = int(
        width
        * vertical_line_a_pct
        / 100
    )

    vertical_line_b = int(
        width
        * vertical_line_b_pct
        / 100
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
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

    vertical_crossings = {}
    horizontal_crossings = {}

    vehicle_speeds = {}
    vehicle_directions = {}

    vehicle_plates = {}

    ocr_attempts = {}
    last_ocr_frame = {}

    seen_vehicle_ids = set()
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

        clean_frame = frame.copy()

        results = model.track(
            source=clean_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # =================================================
        # HORIZONTAL LINES
        # Vehicles moving vertically cross these.
        # =================================================

        cv2.line(
            frame,
            (
                0,
                horizontal_line_a,
            ),
            (
                width,
                horizontal_line_a,
            ),
            (
                255,
                255,
                0,
            ),
            2,
        )

        cv2.line(
            frame,
            (
                0,
                horizontal_line_b,
            ),
            (
                width,
                horizontal_line_b,
            ),
            (
                255,
                0,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Horizontal A "
                f"({horizontal_line_a_pct:.0f}%)"
            ),
            (
                10,
                max(
                    25,
                    horizontal_line_a - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                255,
                255,
                0,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Horizontal B "
                f"({horizontal_line_b_pct:.0f}%)"
            ),
            (
                10,
                max(
                    25,
                    horizontal_line_b - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                255,
                0,
                255,
            ),
            2,
        )

        # =================================================
        # VERTICAL LINES
        # Vehicles moving horizontally cross these.
        # =================================================

        cv2.line(
            frame,
            (
                vertical_line_a,
                0,
            ),
            (
                vertical_line_a,
                height,
            ),
            (
                0,
                255,
                255,
            ),
            2,
        )

        cv2.line(
            frame,
            (
                vertical_line_b,
                0,
            ),
            (
                vertical_line_b,
                height,
            ),
            (
                0,
                165,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Vertical A "
                f"({vertical_line_a_pct:.0f}%)"
            ),
            (
                vertical_line_a + 5,
                25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                255,
                255,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Vertical B "
                f"({vertical_line_b_pct:.0f}%)"
            ),
            (
                vertical_line_b + 5,
                50,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                165,
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

                seen_vehicle_ids.add(
                    vehicle_id
                )

                center_x = int(
                    (
                        x1
                        + x2
                    )
                    / 2
                )

                bottom_y = y2

                # =========================================
                # SPEED MEASUREMENT
                # =========================================

                if (
                    vehicle_id
                    in previous_positions
                    and
                    vehicle_id
                    not in vehicle_speeds
                ):

                    (
                        previous_x,
                        previous_y,
                    ) = (
                        previous_positions[
                            vehicle_id
                        ]
                    )

                    # -------------------------------------
                    # Vertical vehicle movement
                    # -------------------------------------

                    crossed_horizontal_a = (
                        crossed_line(
                            previous_y,
                            bottom_y,
                            horizontal_line_a,
                        )
                    )

                    crossed_horizontal_b = (
                        crossed_line(
                            previous_y,
                            bottom_y,
                            horizontal_line_b,
                        )
                    )

                    vertical_elapsed = (
                        process_crossing_pair(
                            vertical_crossings,
                            vehicle_id,
                            crossed_horizontal_a,
                            crossed_horizontal_b,
                            current_time,
                        )
                    )

                    # -------------------------------------
                    # Horizontal vehicle movement
                    # -------------------------------------

                    crossed_vertical_a = (
                        crossed_line(
                            previous_x,
                            center_x,
                            vertical_line_a,
                        )
                    )

                    crossed_vertical_b = (
                        crossed_line(
                            previous_x,
                            center_x,
                            vertical_line_b,
                        )
                    )

                    horizontal_elapsed = (
                        process_crossing_pair(
                            horizontal_crossings,
                            vehicle_id,
                            crossed_vertical_a,
                            crossed_vertical_b,
                            current_time,
                        )
                    )

                    elapsed_time = None
                    direction = None

                    if (
                        vertical_elapsed
                        is not None
                    ):

                        elapsed_time = (
                            vertical_elapsed
                        )

                        direction = (
                            "VERTICAL"
                        )

                    if (
                        horizontal_elapsed
                        is not None
                    ):

                        if (
                            elapsed_time is None
                            or
                            horizontal_elapsed
                            < elapsed_time
                        ):

                            elapsed_time = (
                                horizontal_elapsed
                            )

                            direction = (
                                "HORIZONTAL"
                            )

                    if (
                        elapsed_time is not None
                        and elapsed_time > 0
                    ):

                        speed = (
                            calculate_speed(
                                calibrated_distance_meters,
                                elapsed_time,
                            )
                        )

                        vehicle_speeds[
                            vehicle_id
                        ] = speed

                        vehicle_directions[
                            vehicle_id
                        ] = direction

                previous_positions[
                    vehicle_id
                ] = (
                    center_x,
                    bottom_y,
                )

                # =========================================
                # NUMBER PLATE RECOGNITION
                # =========================================

                if (
                    vehicle_id
                    not in vehicle_plates
                ):

                    attempts = (
                        ocr_attempts.get(
                            vehicle_id,
                            0,
                        )
                    )

                    previous_ocr_frame = (
                        last_ocr_frame.get(
                            vehicle_id,
                            -100,
                        )
                    )

                    should_try_ocr = (
                        attempts < 5
                        and
                        frame_number
                        - previous_ocr_frame
                        >= 10
                    )

                    if should_try_ocr:

                        vehicle_crop = (
                            crop_vehicle(
                                clean_frame,
                                x1,
                                y1,
                                x2,
                                y2,
                            )
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
                            ) = (
                                detect_and_read_plate(
                                    vehicle_crop
                                )
                            )

                            ocr_attempts[
                                vehicle_id
                            ] = (
                                attempts
                                + 1
                            )

                            last_ocr_frame[
                                vehicle_id
                            ] = (
                                frame_number
                            )

                            if plate_text:

                                vehicle_plates[
                                    vehicle_id
                                ] = (
                                    plate_text
                                )

                                print(
                                    f"Plate detected: "
                                    f"Vehicle {vehicle_id} "
                                    f"-> {plate_text} "
                                    f"({plate_confidence:.2f})"
                                )

                # =========================================
                # VEHICLE DISPLAY
                # =========================================

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

                    direction = (
                        vehicle_directions.get(
                            vehicle_id,
                            "",
                        )
                    )

                    label += (
                        f" | {speed:.1f} km/h"
                        f" | {direction}"
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
                        " | "
                        + vehicle_plates[
                            vehicle_id
                        ]
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
                    0.45,
                    box_color,
                    2,
                )

                # =========================================
                # LOG COMPLETED MEASUREMENT
                # =========================================

                if (
                    vehicle_id
                    in vehicle_speeds
                    and
                    vehicle_id
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
                        or
                        attempts >= 5
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

                        direction = (
                            vehicle_directions.get(
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
                                    clean_frame.copy(),
                                    vehicle_id,
                                    plate,
                                    speed,
                                )
                            )

                        records.append(
                            {
                                "vehicle_id":
                                    vehicle_id,

                                "number_plate":
                                    plate,

                                "movement_direction":
                                    direction,

                                "speed_kmph":
                                    round(
                                        speed,
                                        2,
                                    ),

                                "speed_limit_kmph":
                                    round(
                                        speed_limit_kmph,
                                        2,
                                    ),

                                "timestamp":
                                    datetime.now()
                                    .strftime(
                                        "%Y-%m-%d "
                                        "%H:%M:%S"
                                    ),

                                "violation_status":
                                    violation_status,

                                "evidence_image":
                                    evidence_image,
                            }
                        )

                        logged_vehicles.add(
                            vehicle_id
                        )

        tracked_count = len(
            seen_vehicle_ids
        )

        measured_count = len(
            vehicle_speeds
        )

        plate_count = len(
            vehicle_plates
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
                520,
                140,
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
                f"Tracked: {tracked_count}"
                f" | Measured: "
                f"{measured_count}"
            ),
            (
                20,
                95,
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
                f"Plates: {plate_count}"
                f" | Overspeed: "
                f"{overspeed_count}"
            ),
            (
                20,
                125,
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

            progress_callback(
                min(
                    frame_number
                    / total_frames,
                    1.0,
                )
            )

    cap.release()
    writer.release()

    # =====================================================
    # FALLBACK LOGGING
    # =====================================================

    for (
        vehicle_id,
        speed,
    ) in vehicle_speeds.items():

        if (
            vehicle_id
            in logged_vehicles
        ):
            continue

        plate = (
            vehicle_plates.get(
                vehicle_id,
                "UNKNOWN",
            )
        )

        direction = (
            vehicle_directions.get(
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

        records.append(
            {
                "vehicle_id":
                    vehicle_id,

                "number_plate":
                    plate,

                "movement_direction":
                    direction,

                "speed_kmph":
                    round(
                        speed,
                        2,
                    ),

                "speed_limit_kmph":
                    round(
                        speed_limit_kmph,
                        2,
                    ),

                "timestamp":
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d "
                        "%H:%M:%S"
                    ),

                "violation_status":
                    violation_status,

                "evidence_image":
                    "",
            }
        )

        logged_vehicles.add(
            vehicle_id
        )

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

    vertical_measurements = sum(
        1
        for direction
        in vehicle_directions.values()
        if direction
        == "VERTICAL"
    )

    horizontal_measurements = sum(
        1
        for direction
        in vehicle_directions.values()
        if direction
        == "HORIZONTAL"
    )

    summary = {

        "tracked_vehicles":
            len(
                seen_vehicle_ids
            ),

        "measured_vehicles":
            len(
                vehicle_speeds
            ),

        "plates_detected":
            len(
                vehicle_plates
            ),

        "total_records":
            len(
                dataframe
            ),

        "overspeed_violations":
            sum(
                1
                for speed
                in vehicle_speeds.values()
                if speed
                > speed_limit_kmph
            ),

        "vertical_measurements":
            vertical_measurements,

        "horizontal_measurements":
            horizontal_measurements,

        "horizontal_line_a_pct":
            horizontal_line_a_pct,

        "horizontal_line_b_pct":
            horizontal_line_b_pct,

        "vertical_line_a_pct":
            vertical_line_a_pct,

        "vertical_line_b_pct":
            vertical_line_b_pct,

        "output_video":
            str(
                output_video
            ),

        "csv_file":
            str(
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