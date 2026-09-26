# app/streamlit_app.py

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.web_pipeline import process_video


st.set_page_config(
    page_title="AI Speed Detection and ANPR",
    layout="wide",
)


INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "input"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
)

VIOLATION_DIR = (
    OUTPUT_DIR
    / "violations"
)

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIOLATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None


st.title(
    "AI-Based Speed Detection and Number Plate Recognition"
)

st.write(
    """
    Upload a traffic video to detect and track vehicles,
    estimate speed, recognize registration plates,
    and identify overspeed violations.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Detection Settings"
    )

    speed_limit = st.number_input(
        "Speed Limit (km/h)",
        min_value=10.0,
        max_value=200.0,
        value=50.0,
        step=5.0,
    )

    calibrated_distance = st.number_input(
        "Distance Between Speed Lines (meters)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

    st.divider()

    st.header(
        "Speed Line Positions"
    )

    st.caption(
        "Move these sliders so the speed lines "
        "cross the actual traffic path."
    )

    horizontal_line_a = st.slider(
        "Horizontal Line A (% height)",
        min_value=5,
        max_value=90,
        value=45,
        step=1,
    )

    horizontal_line_b = st.slider(
        "Horizontal Line B (% height)",
        min_value=10,
        max_value=95,
        value=60,
        step=1,
    )

    vertical_line_a = st.slider(
        "Vertical Line A (% width)",
        min_value=5,
        max_value=90,
        value=40,
        step=1,
    )

    vertical_line_b = st.slider(
        "Vertical Line B (% width)",
        min_value=10,
        max_value=95,
        value=60,
        step=1,
    )

    valid_lines = True

    if (
        horizontal_line_a
        >= horizontal_line_b
    ):

        st.error(
            "Horizontal Line A must be "
            "before Horizontal Line B."
        )

        valid_lines = False

    if (
        vertical_line_a
        >= vertical_line_b
    ):

        st.error(
            "Vertical Line A must be "
            "before Vertical Line B."
        )

        valid_lines = False

    st.divider()

    st.subheader(
        "Technology"
    )

    st.write(
        "Vehicle Detection: YOLO"
    )

    st.write(
        "Tracking: ByteTrack"
    )

    st.write(
        "Plate Detection: YOLO"
    )

    st.write(
        "OCR: EasyOCR"
    )

    st.write(
        "Computer Vision: OpenCV"
    )

    st.write(
        "Interface: Streamlit"
    )

    st.divider()

    st.caption(
        "Speed values remain estimates until "
        "the physical road distance is calibrated."
    )


# =========================================================
# VIDEO UPLOAD
# =========================================================

st.subheader(
    "Upload Traffic Video"
)

uploaded_video = st.file_uploader(
    "Choose a traffic video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_video is not None:

    st.success(
        f"Uploaded: {uploaded_video.name}"
    )

    st.video(
        uploaded_video
    )

    suffix = (
        Path(
            uploaded_video.name
        )
        .suffix
        .lower()
    )

    input_video_path = (
        INPUT_DIR
        / f"uploaded_traffic{suffix}"
    )

    output_video_path = (
        OUTPUT_DIR
        / "processed_traffic.mp4"
    )

    with open(
        input_video_path,
        "wb",
    ) as file:

        file.write(
            uploaded_video.getbuffer()
        )

    st.write(
        f"Video size: "
        f"{uploaded_video.size / (1024 * 1024):.2f} MB"
    )

    process_button = st.button(
        "Process Traffic Video",
        type="primary",
        use_container_width=True,
        disabled=not valid_lines,
    )

    if process_button:

        progress_bar = st.progress(
            0
        )

        status_message = st.empty()

        def update_progress(
            progress,
        ):

            progress_value = int(
                progress
                * 100
            )

            progress_bar.progress(
                progress_value
            )

            status_message.write(
                f"Processing video: "
                f"{progress_value}%"
            )

        try:

            with st.spinner(
                "Running vehicle detection, tracking, "
                "speed estimation and ANPR..."
            ):

                dataframe, summary = process_video(

                    input_video=(
                        input_video_path
                    ),

                    output_video=(
                        output_video_path
                    ),

                    calibrated_distance_meters=(
                        calibrated_distance
                    ),

                    speed_limit_kmph=(
                        speed_limit
                    ),

                    horizontal_line_a_pct=(
                        float(
                            horizontal_line_a
                        )
                    ),

                    horizontal_line_b_pct=(
                        float(
                            horizontal_line_b
                        )
                    ),

                    vertical_line_a_pct=(
                        float(
                            vertical_line_a
                        )
                    ),

                    vertical_line_b_pct=(
                        float(
                            vertical_line_b
                        )
                    ),

                    progress_callback=(
                        update_progress
                    ),
                )

            st.session_state.processed_data = (
                dataframe
            )

            st.session_state.summary = (
                summary
            )

            st.session_state.processed_video = (
                str(
                    output_video_path
                )
            )

            progress_bar.progress(
                100
            )

            status_message.success(
                "Video processing completed."
            )

        except Exception as error:

            st.error(
                f"Processing failed: {error}"
            )


st.divider()


# =========================================================
# RESULTS
# =========================================================

data = (
    st.session_state
    .processed_data
)

summary = (
    st.session_state
    .summary
)


if (
    data is not None
    and summary is not None
):

    st.header(
        "Detection Results"
    )

    col1, col2, col3, col4, col5 = (
        st.columns(
            5
        )
    )

    with col1:

        st.metric(
            "Tracked Vehicles",
            summary.get(
                "tracked_vehicles",
                0,
            ),
        )

    with col2:

        st.metric(
            "Measured Vehicles",
            summary.get(
                "measured_vehicles",
                0,
            ),
        )

    with col3:

        st.metric(
            "Plates Detected",
            summary.get(
                "plates_detected",
                0,
            ),
        )

    with col4:

        st.metric(
            "Vehicles Logged",
            summary.get(
                "total_records",
                0,
            ),
        )

    with col5:

        st.metric(
            "Overspeed Violations",
            summary.get(
                "overspeed_violations",
                0,
            ),
        )

    st.caption(
        "Tracked Vehicles = unique YOLO + ByteTrack IDs. "
        "Measured Vehicles = vehicles that successfully "
        "crossed one complete speed-line pair."
    )


    # =====================================================
    # MEASUREMENT DIRECTION
    # =====================================================

    st.subheader(
        "Speed Measurement Direction"
    )

    direction_col1, direction_col2 = (
        st.columns(
            2
        )
    )

    with direction_col1:

        st.metric(
            "Vertical-Traffic Measurements",
            summary.get(
                "vertical_measurements",
                0,
            ),
        )

    with direction_col2:

        st.metric(
            "Horizontal-Traffic Measurements",
            summary.get(
                "horizontal_measurements",
                0,
            ),
        )

    st.caption(
        "Vertical traffic crosses the horizontal lines. "
        "Horizontal traffic crosses the vertical lines."
    )


    # =====================================================
    # LINE SETTINGS USED
    # =====================================================

    with st.expander(
        "Speed Line Configuration Used"
    ):

        st.write(
            "Horizontal Line A:",
            f"{summary.get('horizontal_line_a_pct', 0):.0f}%",
        )

        st.write(
            "Horizontal Line B:",
            f"{summary.get('horizontal_line_b_pct', 0):.0f}%",
        )

        st.write(
            "Vertical Line A:",
            f"{summary.get('vertical_line_a_pct', 0):.0f}%",
        )

        st.write(
            "Vertical Line B:",
            f"{summary.get('vertical_line_b_pct', 0):.0f}%",
        )


    # =====================================================
    # PROCESSED VIDEO
    # =====================================================

    st.subheader(
        "Processed Video"
    )

    processed_video_path = (
        st.session_state
        .processed_video
    )

    if (
        processed_video_path
        and
        Path(
            processed_video_path
        ).exists()
    ):

        st.video(
            processed_video_path
        )

        with open(
            processed_video_path,
            "rb",
        ) as video_file:

            st.download_button(
                label=(
                    "Download Processed Video"
                ),
                data=(
                    video_file.read()
                ),
                file_name=(
                    "processed_traffic.mp4"
                ),
                mime="video/mp4",
                use_container_width=True,
            )

    else:

        st.warning(
            "Processed video unavailable."
        )


    st.divider()


    # =====================================================
    # VEHICLE RECORDS
    # =====================================================

    st.subheader(
        "Vehicle Records"
    )

    if (
        isinstance(
            data,
            pd.DataFrame,
        )
        and
        not data.empty
    ):

        display_columns = [
            "vehicle_id",
            "number_plate",
            "movement_direction",
            "speed_kmph",
            "speed_limit_kmph",
            "timestamp",
            "violation_status",
        ]

        available_columns = [
            column
            for column
            in display_columns
            if column
            in data.columns
        ]

        st.dataframe(
            data[
                available_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No completed speed measurements "
            "were recorded."
        )


    st.divider()


    # =====================================================
    # SPEED ANALYSIS
    # =====================================================

    st.subheader(
        "Speed Analysis"
    )

    if (
        isinstance(
            data,
            pd.DataFrame,
        )
        and
        not data.empty
        and
        "speed_kmph"
        in data.columns
    ):

        chart_data = (
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
            chart_data
        )

    else:

        st.info(
            "No speed measurements available."
        )


    st.divider()


    # =====================================================
    # OVERSPEED
    # =====================================================

    st.subheader(
        "Overspeed Violations"
    )

    if (
        isinstance(
            data,
            pd.DataFrame,
        )
        and
        not data.empty
        and
        "violation_status"
        in data.columns
    ):

        overspeed_data = (
            data[
                data[
                    "violation_status"
                ]
                == "OVERSPEED"
            ]
        )

        if overspeed_data.empty:

            st.success(
                "No overspeed violations detected."
            )

        else:

            st.dataframe(
                overspeed_data,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No overspeed data available."
        )


    st.divider()


    # =====================================================
    # NUMBER PLATE SEARCH
    # =====================================================

    st.subheader(
        "Search Number Plate"
    )

    search_plate = st.text_input(
        "Enter registration number",
        placeholder=(
            "Example: KA01MN1234"
        ),
    )

    if (
        search_plate
        and
        isinstance(
            data,
            pd.DataFrame,
        )
        and
        not data.empty
        and
        "number_plate"
        in data.columns
    ):

        normalized_search = (
            search_plate
            .replace(
                " ",
                "",
            )
            .upper()
        )

        search_results = (
            data[
                data[
                    "number_plate"
                ]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(
                    normalized_search,
                    na=False,
                )
            ]
        )

        if search_results.empty:

            st.warning(
                "No matching vehicle found."
            )

        else:

            st.success(
                f"{len(search_results)} "
                f"record(s) found."
            )

            st.dataframe(
                search_results,
                use_container_width=True,
                hide_index=True,
            )


    st.divider()


    # =====================================================
    # EXPORT
    # =====================================================

    st.subheader(
        "Export Violation Report"
    )

    if (
        isinstance(
            data,
            pd.DataFrame,
        )
        and
        not data.empty
    ):

        csv_data = (
            data
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            label=(
                "Download Violation Report"
            ),
            data=(
                csv_data
            ),
            file_name=(
                "traffic_violation_report.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    else:

        st.info(
            "No report is available yet."
        )


else:

    st.info(
        "Upload a traffic video and select "
        "'Process Traffic Video' to begin."
    )