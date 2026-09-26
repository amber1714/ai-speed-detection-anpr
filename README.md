# AI-Based Speed Detection and Automatic Number Plate Recognition

An end-to-end computer vision system for detecting vehicles, tracking them across frames, estimating speed, recognizing number plates, detecting overspeed violations, and visualizing results through a Streamlit dashboard.

## Features

- Vehicle detection using Ultralytics YOLO
- Multi-object tracking using ByteTrack
- Vehicle speed estimation
- Automatic Number Plate Recognition (ANPR)
- EasyOCR-based plate recognition
- Indian number-plate OCR correction
- Overspeed violation detection
- CSV-based violation logging
- Evidence-image capture
- Streamlit analytics dashboard
- Number-plate search
- Downloadable violation reports

## System Pipeline

```text
Traffic Video
      |
      v
Vehicle Detection
      |
      v
Vehicle Tracking
      |
      v
Speed Estimation
      |
      v
Number Plate Detection
      |
      v
OCR
      |
      v
Plate Normalization
      |
      v
Speed Limit Check
      |
      v
Violation Logging
      |
      v
Streamlit Dashboard
```

## Project Structure

```text
ai-speed-detection-anpr/
|
|-- app/
|   `-- streamlit_app.py
|
|-- data/
|   |-- input/
|   `-- output/
|
|-- src/
|   |-- __init__.py
|   |-- vehicle_detector.py
|   |-- vehicle_tracker.py
|   |-- speed_estimator.py
|   |-- plate_reader.py
|   `-- violation_logger.py
|
|-- main.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Technologies

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack
- EasyOCR
- NumPy
- Pandas
- Streamlit
- PyTorch

## Installation

Clone the repository:

```bash
git clone https://github.com/amber1714/ai-speed-detection-anpr.git
cd ai-speed-detection-anpr
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Detection Pipeline

Place a traffic video here:

```text
data/input/traffic.mp4
```

Run:

```bash
python main.py
```

Generated results are stored inside:

```text
data/output/
```

## Run the Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard provides:

- Vehicle records
- Detected speeds
- Overspeed violations
- Number plates
- Evidence images
- Speed charts
- Registration-number search
- Downloadable CSV report

## Speed Estimation

The system estimates vehicle speed based on the time required for a tracked vehicle to cross two virtual road lines.

```text
Speed (m/s) = Distance (m) / Time (s)

Speed (km/h) = Speed (m/s) × 3.6
```

The configured road distance must be calibrated to match the actual real-world distance represented between the virtual lines.

## Number Plate Recognition

The system attempts to locate the registration plate within the detected vehicle and passes it through EasyOCR.

OCR post-processing corrects common character mistakes.

Example:

```text
KAOIMN1234
```

is corrected to:

```text
KA01MN1234
```

Typical corrections include:

```text
O → 0
I → 1
Q → 0
0 → O
1 → I
```

depending on whether a letter or number is expected at that position.

## Violation Logging

The system can record:

- Vehicle ID
- Registration number
- Detected speed
- Speed limit
- Timestamp
- Violation status
- Evidence-image location

## Current Limitations

- Speed accuracy depends on camera positioning and calibration.
- Perspective distortion can affect speed measurements.
- OCR performance depends on plate visibility, resolution, lighting, angle, and motion blur.
- Plate localization currently uses computer-vision heuristics.
- The current system is intended for research and demonstration rather than certified traffic enforcement.

## Future Improvements

- Dedicated YOLO number-plate detection model
- Homography-based road calibration
- More accurate perspective-aware speed estimation
- Live CCTV/RTSP support
- Database integration
- Real-time violation alerts
- Vehicle-type analytics
- Improved Indian plate validation
- Dashboard video upload and processing
- Cloud deployment
- REST API
- Authentication for traffic administrators

## Author

**Amber Rodrigues**

M.Tech Artificial Intelligence

## Disclaimer

This project is intended for educational and research purposes. Speed measurements should not be considered legally certified enforcement measurements without appropriate calibrated equipment, controlled testing, and regulatory approval.