# src/violation_logger.py

from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
VIOLATION_DIR = OUTPUT_DIR / "violations"
CSV_FILE = OUTPUT_DIR / "violations.csv"


def initialize_violation_storage():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIOLATION_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        dataframe = pd.DataFrame(
            columns=[
                "vehicle_id",
                "number_plate",
                "speed_kmph",
                "speed_limit_kmph",
                "timestamp",
                "violation_status",
                "evidence_image",
            ]
        )

        dataframe.to_csv(
            CSV_FILE,
            index=False,
        )


def sanitize_plate_text(plate_text):
    if plate_text is None:
        return "UNKNOWN"

    plate_text = str(plate_text).strip().upper()

    if not plate_text:
        return "UNKNOWN"

    return plate_text


def save_evidence_image(
    frame,
    vehicle_id,
    plate_text,
    speed,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    plate_text = sanitize_plate_text(
        plate_text
    )

    filename = (
        f"vehicle_{vehicle_id}_"
        f"{plate_text}_"
        f"{speed:.1f}kmph_"
        f"{timestamp}.jpg"
    )

    evidence_path = (
        VIOLATION_DIR / filename
    )

    success = cv2.imwrite(
        str(evidence_path),
        frame,
    )

    if not success:
        return None

    return evidence_path


def log_violation(
    vehicle_id,
    plate_text,
    speed,
    speed_limit,
    frame=None,
):
    initialize_violation_storage()

    plate_text = sanitize_plate_text(
        plate_text
    )

    speed = float(speed)
    speed_limit = float(speed_limit)

    is_overspeeding = (
        speed > speed_limit
    )

    violation_status = (
        "OVERSPEED"
        if is_overspeeding
        else "NORMAL"
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    evidence_path = None

    if (
        is_overspeeding
        and frame is not None
    ):
        evidence_path = save_evidence_image(
            frame=frame,
            vehicle_id=vehicle_id,
            plate_text=plate_text,
            speed=speed,
        )

    record = {
        "vehicle_id": vehicle_id,
        "number_plate": plate_text,
        "speed_kmph": round(speed, 2),
        "speed_limit_kmph": round(
            speed_limit,
            2,
        ),
        "timestamp": timestamp,
        "violation_status": violation_status,
        "evidence_image": (
            str(evidence_path)
            if evidence_path
            else ""
        ),
    }

    dataframe = pd.DataFrame(
        [record]
    )

    dataframe.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False,
    )

    return record


def violation_already_logged(
    vehicle_id,
):
    initialize_violation_storage()

    try:
        dataframe = pd.read_csv(
            CSV_FILE
        )

        if dataframe.empty:
            return False

        vehicle_ids = (
            dataframe["vehicle_id"]
            .astype(str)
            .tolist()
        )

        return (
            str(vehicle_id)
            in vehicle_ids
        )

    except Exception:
        return False


def get_all_violations():
    initialize_violation_storage()

    try:
        dataframe = pd.read_csv(
            CSV_FILE
        )

        return dataframe

    except Exception:
        return pd.DataFrame()


def get_overspeed_violations():
    dataframe = get_all_violations()

    if dataframe.empty:
        return dataframe

    return dataframe[
        dataframe[
            "violation_status"
        ]
        == "OVERSPEED"
    ]


def clear_violation_log():
    initialize_violation_storage()

    dataframe = pd.DataFrame(
        columns=[
            "vehicle_id",
            "number_plate",
            "speed_kmph",
            "speed_limit_kmph",
            "timestamp",
            "violation_status",
            "evidence_image",
        ]
    )

    dataframe.to_csv(
        CSV_FILE,
        index=False,
    )


def test_logger():
    initialize_violation_storage()

    test_record = log_violation(
        vehicle_id=1,
        plate_text="KA01MN1234",
        speed=72.5,
        speed_limit=50,
        frame=None,
    )

    print(
        "\nSaved record:"
    )

    print(
        test_record
    )

    print(
        "\nViolation file:"
    )

    print(
        CSV_FILE
    )

    print(
        "\nCurrent violations:"
    )

    print(
        get_all_violations()
    )


if __name__ == "__main__":
    test_logger()