from pathlib import Path

import cv2
from ultralytics import YOLO

from src.plate_reader import detect_and_read_plate
from src.violation_logger import (
    initialize_violation_storage,
    log_violation,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "traffic.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "final_speed_anpr_output.mp4"
)


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_NAME = "yolo26n.pt"

# COCO vehicle classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

CONFIDENCE_THRESHOLD = 0.35


# =========================================================
# SPEED SETTINGS
# =========================================================

# Temporary test value.
# Later we will calibrate this accurately.
CALIBRATED_DISTANCE_METERS = 10.0

SPEED_LIMIT_KMPH = 50.0


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def crossed_line(
    previous_y,
    current_y,
    line_y,
):
    return (
        previous_y < line_y <= current_y
        or previous_y > line_y >= current_y
    )


def calculate_speed(
    distance_meters,
    elapsed_seconds,
):
    if elapsed_seconds <= 0:
        return 0.0

    speed_mps = (
        distance_meters
        / elapsed_seconds
    )

    speed_kmph = (
        speed_mps
        * 3.6
    )

    return speed_kmph


def crop_vehicle(
    frame,
    x1,
    y1,
    x2,
    y2,
):
    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        min(
            x1,
            frame_width - 1,
        ),
    )

    x2 = max(
        0,
        min(
            x2,
            frame_width,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            frame_height - 1,
        ),
    )

    y2 = max(
        0,
        min(
            y2,
            frame_height,
        ),
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


def draw_label(
    frame,
    text,
    x,
    y,
    color,
):
    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.55
    thickness = 2

    (
        text_width,
        text_height,
    ), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    label_y = max(
        y,
        text_height + 12,
    )

    cv2.rectangle(
        frame,
        (
            x,
            label_y
            - text_height
            - 10,
        ),
        (
            x + text_width + 10,
            label_y + 5,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x + 5,
            label_y - 3,
        ),
        font,
        font_scale,
        (
            255,
            255,
            255,
        ),
        thickness,
        cv2.LINE_AA,
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_system():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_violation_storage()

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            "\nTraffic video not found.\n"
            f"Expected location:\n"
            f"{INPUT_VIDEO}"
        )

    print(
        "\nLoading YOLO model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    cap = cv2.VideoCapture(
        str(
            INPUT_VIDEO
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open traffic video."
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

    # Virtual measurement lines
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
        str(
            OUTPUT_VIDEO
        ),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    # =====================================================
    # VEHICLE STATE STORAGE
    # =====================================================

    previous_positions = {}

    first_crossings = {}

    vehicle_speeds = {}

    vehicle_plates = {}

    plate_confidences = {}

    logged_vehicles = set()

    ocr_attempts = {}

    frame_number = 0

    print(
        "\nAI Speed Detection + ANPR started."
    )

    print(
        "\nPress Q to stop the program."
    )

    # =====================================================
    # VIDEO LOOP
    # =====================================================

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        frame_number += 1

        current_time = (
            frame_number
            / fps
        )

        # -------------------------------------------------
        # YOLO VEHICLE TRACKING
        # -------------------------------------------------

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        # -------------------------------------------------
        # DRAW SPEED LINES
        # -------------------------------------------------

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
                line_a - 10,
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
                line_b - 10,
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

        boxes = (
            results[0].boxes
        )

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

            confidences = (
                boxes.conf
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
                confidence,
                vehicle_id,
            ) in zip(
                xyxy_boxes,
                class_ids,
                confidences,
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

                # Bottom center is better for road crossing
                bottom_y = y2

                # -----------------------------------------
                # SPEED ESTIMATION
                # -----------------------------------------

                if (
                    vehicle_id
                    in previous_positions
                ):

                    previous_y = (
                        previous_positions[
                            vehicle_id
                        ]
                    )

                    crossed_a = (
                        crossed_line(
                            previous_y,
                            bottom_y,
                            line_a,
                        )
                    )

                    crossed_b = (
                        crossed_line(
                            previous_y,
                            bottom_y,
                            line_b,
                        )
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
                                    "time": (
                                        current_time
                                    ),
                                }

                            elif crossed_b:

                                first_crossings[
                                    vehicle_id
                                ] = {
                                    "line": "B",
                                    "time": (
                                        current_time
                                    ),
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

                            measurement_done = (
                                False
                            )

                            if (
                                first_line
                                == "A"
                                and crossed_b
                            ):

                                measurement_done = (
                                    True
                                )

                            elif (
                                first_line
                                == "B"
                                and crossed_a
                            ):

                                measurement_done = (
                                    True
                                )

                            if measurement_done:

                                elapsed_time = (
                                    current_time
                                    - first_time
                                )

                                speed = (
                                    calculate_speed(
                                        CALIBRATED_DISTANCE_METERS,
                                        elapsed_time,
                                    )
                                )

                                vehicle_speeds[
                                    vehicle_id
                                ] = speed

                                print(
                                    f"\nVehicle "
                                    f"{vehicle_id} "
                                    f"speed: "
                                    f"{speed:.2f} km/h"
                                )

                previous_positions[
                    vehicle_id
                ] = bottom_y

                # -----------------------------------------
                # NUMBER PLATE RECOGNITION
                # -----------------------------------------

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

                        vehicle_crop = (
                            crop_vehicle(
                                frame,
                                x1,
                                y1,
                                x2,
                                y2,
                            )
                        )

                        if (
                            vehicle_crop
                            is not None
                            and
                            vehicle_crop.size
                            > 0
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

                            if plate_text:

                                vehicle_plates[
                                    vehicle_id
                                ] = (
                                    plate_text
                                )

                                plate_confidences[
                                    vehicle_id
                                ] = (
                                    plate_confidence
                                )

                                print(
                                    f"\nVehicle "
                                    f"{vehicle_id} "
                                    f"plate: "
                                    f"{plate_text}"
                                )

                # -----------------------------------------
                # DISPLAY VEHICLE INFORMATION
                # -----------------------------------------

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
                        > SPEED_LIMIT_KMPH
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

                draw_label(
                    frame,
                    label,
                    x1,
                    y1 - 5,
                    box_color,
                )

                # -----------------------------------------
                # LOG COMPLETED VEHICLE
                # -----------------------------------------

                if (
                    vehicle_id
                    in vehicle_speeds
                    and vehicle_id
                    not in logged_vehicles
                ):

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

                    # Give OCR several frames before
                    # logging an UNKNOWN plate
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

                        record = (
                            log_violation(
                                vehicle_id=(
                                    vehicle_id
                                ),
                                plate_text=(
                                    plate
                                ),
                                speed=(
                                    speed
                                ),
                                speed_limit=(
                                    SPEED_LIMIT_KMPH
                                ),
                                frame=(
                                    frame.copy()
                                ),
                            )
                        )

                        logged_vehicles.add(
                            vehicle_id
                        )

                        print(
                            "\nLogged vehicle:"
                        )

                        print(
                            record
                        )

        # -------------------------------------------------
        # DASHBOARD OVERLAY
        # -------------------------------------------------

        overspeed_count = sum(
            1
            for speed
            in vehicle_speeds.values()
            if speed
            > SPEED_LIMIT_KMPH
        )

        cv2.rectangle(
            frame,
            (
                10,
                10,
            ),
            (
                420,
                120,
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
                f"{SPEED_LIMIT_KMPH:.0f} km/h"
            ),
            (
                20,
                65,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
                f"Measured Vehicles: "
                f"{len(vehicle_speeds)}"
            ),
            (
                20,
                90,
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
                f"Overspeed: "
                f"{overspeed_count}"
            ),
            (
                220,
                90,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                255,
            ),
            2,
        )

        # -------------------------------------------------
        # SAVE OUTPUT
        # -------------------------------------------------

        writer.write(
            frame
        )

        cv2.imshow(
            "AI Speed Detection and ANPR",
            frame,
        )

        # -------------------------------------------------
        # TERMINAL PROGRESS
        # -------------------------------------------------

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

        # Q = quit
        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print(
        "\n\n================================="
    )

    print(
        "PROCESSING COMPLETED"
    )

    print(
        "================================="
    )

    print(
        f"\nOutput video:\n"
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"\nVehicles with measured speed: "
        f"{len(vehicle_speeds)}"
    )

    print(
        f"Vehicles with detected plates: "
        f"{len(vehicle_plates)}"
    )

    print(
        f"Vehicles logged: "
        f"{len(logged_vehicles)}"
    )

    print(
        "\nDetected vehicles:"
    )

    for vehicle_id, speed in (
        vehicle_speeds.items()
    ):

        plate = (
            vehicle_plates.get(
                vehicle_id,
                "UNKNOWN",
            )
        )

        status = (
            "OVERSPEED"
            if speed
            > SPEED_LIMIT_KMPH
            else "NORMAL"
        )

        print(
            f"ID: {vehicle_id}"
            f" | Plate: {plate}"
            f" | Speed: {speed:.2f} km/h"
            f" | Status: {status}"
        )


if __name__ == "__main__":
    run_system()