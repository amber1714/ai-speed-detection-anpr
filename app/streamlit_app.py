# app/streamlit_app.py

from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Speed Detection & ANPR",
    page_icon="🚗",
    layout="wide",
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

CSV_FILE = OUTPUT_DIR / "violations.csv"

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "final_speed_anpr_output.mp4"
)

VIOLATION_DIR = (
    OUTPUT_DIR
    / "violations"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(ttl=2)
def load_data():
    if not CSV_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(CSV_FILE)

    except Exception:
        return pd.DataFrame()


data = load_data()


# =========================================================
# HEADER
# =========================================================

st.title(
    "AI-Based Speed Detection & Number Plate Recognition"
)

st.write(
    """
    Intelligent traffic-monitoring system using vehicle detection,
    tracking, speed estimation, ANPR and automated violation logging.
    """
)

st.divider()


# =========================================================
# METRICS
# =========================================================

if not data.empty:

    total_vehicles = len(data)

    overspeed_data = data[
        data["violation_status"]
        == "OVERSPEED"
    ]

    overspeed_count = len(
        overspeed_data
    )

    normal_count = (
        total_vehicles
        - overspeed_count
    )

    valid_plates = data[
        data["number_plate"]
        .fillna("UNKNOWN")
        != "UNKNOWN"
    ]

    detected_plates = len(
        valid_plates
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        st.metric(
            "Vehicles Logged",
            total_vehicles,
        )

    with col2:
        st.metric(
            "Overspeed Violations",
            overspeed_count,
        )

    with col3:
        st.metric(
            "Normal Vehicles",
            normal_count,
        )

    with col4:
        st.metric(
            "Number Plates Detected",
            detected_plates,
        )

else:

    st.info(
        "No vehicle records are available yet. Run main.py first."
    )


# =========================================================
# VIDEO SECTION
# =========================================================

st.subheader(
    "Processed Traffic Video"
)

if OUTPUT_VIDEO.exists():

    with open(
        OUTPUT_VIDEO,
        "rb",
    ) as video_file:

        video_bytes = (
            video_file.read()
        )

        st.video(
            video_bytes
        )

else:

    st.warning(
        "Processed video not found."
    )


st.divider()


# =========================================================
# VEHICLE RECORDS
# =========================================================

st.subheader(
    "Vehicle Detection Records"
)

if not data.empty:

    display_columns = [
        "vehicle_id",
        "number_plate",
        "speed_kmph",
        "speed_limit_kmph",
        "timestamp",
        "violation_status",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in data.columns
    ]

    st.dataframe(
        data[
            available_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.write(
        "No records available."
    )


st.divider()


# =========================================================
# OVERSPEED VIOLATIONS
# =========================================================

st.subheader(
    "Overspeed Violations"
)

if not data.empty:

    overspeed_data = data[
        data[
            "violation_status"
        ]
        == "OVERSPEED"
    ]

    if not overspeed_data.empty:

        for _, row in (
            overspeed_data.iterrows()
        ):

            vehicle_id = row.get(
                "vehicle_id",
                "Unknown",
            )

            number_plate = row.get(
                "number_plate",
                "UNKNOWN",
            )

            speed = row.get(
                "speed_kmph",
                0,
            )

            speed_limit = row.get(
                "speed_limit_kmph",
                0,
            )

            timestamp = row.get(
                "timestamp",
                "",
            )

            evidence_path = row.get(
                "evidence_image",
                "",
            )

            with st.container(
                border=True
            ):

                col1, col2 = (
                    st.columns(
                        [2, 1]
                    )
                )

                with col1:

                    st.write(
                        f"Vehicle ID: {vehicle_id}"
                    )

                    st.write(
                        f"Number Plate: {number_plate}"
                    )

                    st.write(
                        f"Detected Speed: {speed} km/h"
                    )

                    st.write(
                        f"Speed Limit: {speed_limit} km/h"
                    )

                    st.write(
                        f"Timestamp: {timestamp}"
                    )

                    st.error(
                        "OVERSPEED VIOLATION"
                    )

                with col2:

                    if (
                        isinstance(
                            evidence_path,
                            str,
                        )
                        and evidence_path.strip()
                    ):

                        image_path = Path(
                            evidence_path
                        )

                        if image_path.exists():

                            st.image(
                                str(
                                    image_path
                                ),
                                caption=(
                                    f"Evidence - "
                                    f"{number_plate}"
                                ),
                                use_container_width=True,
                            )

                        else:

                            st.write(
                                "Evidence image unavailable."
                            )

                    else:

                        st.write(
                            "No evidence image."
                        )

    else:

        st.success(
            "No overspeed violations detected."
        )


st.divider()


# =========================================================
# SPEED ANALYSIS
# =========================================================

st.subheader(
    "Speed Analysis"
)

if (
    not data.empty
    and "speed_kmph"
    in data.columns
):

    speed_data = (
        data[
            [
                "vehicle_id",
                "speed_kmph",
            ]
        ]
        .set_index(
            "vehicle_id"
        )
    )

    st.bar_chart(
        speed_data
    )


# =========================================================
# NUMBER PLATE SEARCH
# =========================================================

st.divider()

st.subheader(
    "Search Number Plate"
)

search_plate = st.text_input(
    "Enter registration number",
    placeholder="Example: KA01MN1234",
)

if (
    search_plate
    and not data.empty
):

    search_plate = (
        search_plate
        .replace(
            " ",
            "",
        )
        .upper()
    )

    results = data[
        data[
            "number_plate"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains(
            search_plate,
            na=False,
        )
    ]

    if not results.empty:

        st.success(
            f"{len(results)} record(s) found."
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "No matching vehicle found."
        )


# =========================================================
# DOWNLOAD CSV
# =========================================================

st.divider()

st.subheader(
    "Export Violation Report"
)

if not data.empty:

    csv_data = (
        data.to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label="Download Violation Report",
        data=csv_data,
        file_name="traffic_violation_report.csv",
        mime="text/csv",
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "Vehicle Detection: YOLO"
    )

    st.write(
        "Tracking: ByteTrack"
    )

    st.write(
        "OCR: EasyOCR"
    )

    st.write(
        "Computer Vision: OpenCV"
    )

    st.write(
        "Dashboard: Streamlit"
    )

    st.divider()

    if st.button(
        "Refresh Dashboard"
    ):

        st.cache_data.clear()

        st.rerun()