# src/plate_reader.py

import re
from pathlib import Path

import cv2
import easyocr


# ---------------------------------------------------------
# INITIALIZE OCR
# ---------------------------------------------------------

reader = easyocr.Reader(
    ["en"],
    gpu=False,
)


# ---------------------------------------------------------
# CLEAN OCR TEXT
# ---------------------------------------------------------

def clean_plate_text(text):
    """
    Remove spaces and special characters.

    Example:
    "KA 01 MN 1234"
    becomes
    "KA01MN1234"
    """

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    return text


# ---------------------------------------------------------
# FIX COMMON OCR ERRORS
# ---------------------------------------------------------

def normalize_indian_plate(text):
    """
    Correct common OCR mistakes in Indian number plates.

    Example:

    KAOIMN1234
    becomes
    KA01MN1234
    """

    text = clean_plate_text(text)

    if len(text) < 8:
        return text

    characters = list(text)

    # When OCR reads a number as a letter
    digit_replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    # When OCR reads a letter as a number
    letter_replacements = {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
    }

    # -------------------------------------------------
    # Indian registration structure
    #
    # Example:
    #
    # KA01MN1234
    #
    # KA = State code
    # 01 = RTO number
    # MN = Series letters
    # 1234 = Registration number
    # -------------------------------------------------

    # First two positions should be LETTERS
    for i in range(
        0,
        min(2, len(characters)),
    ):

        characters[i] = letter_replacements.get(
            characters[i],
            characters[i],
        )

    # Positions 2 and 3 should be DIGITS
    for i in range(
        2,
        min(4, len(characters)),
    ):

        characters[i] = digit_replacements.get(
            characters[i],
            characters[i],
        )

    # Last four characters should be DIGITS
    start_last_digits = max(
        4,
        len(characters) - 4,
    )

    for i in range(
        start_last_digits,
        len(characters),
    ):

        characters[i] = digit_replacements.get(
            characters[i],
            characters[i],
        )

    # Middle characters should generally be LETTERS
    for i in range(
        4,
        start_last_digits,
    ):

        characters[i] = letter_replacements.get(
            characters[i],
            characters[i],
        )

    corrected_text = "".join(
        characters
    )

    return corrected_text


# ---------------------------------------------------------
# VALIDATE INDIAN NUMBER PLATE
# ---------------------------------------------------------

def is_possible_indian_plate(text):
    """
    Basic validation of Indian registration numbers.

    Examples:

    KA01MN1234
    TN38CD5678
    GA08A1234
    KL07AB1234
    """

    patterns = [

        # Example:
        # KA01MN1234
        r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$",

        # Some registrations may contain
        # one-digit RTO numbers
        r"^[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{4}$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text,
        ):

            return True

    return False


# ---------------------------------------------------------
# PREPROCESS PLATE IMAGE
# ---------------------------------------------------------

def preprocess_plate(image):
    """
    Improve the number plate image
    before sending it to OCR.
    """

    if (
        image is None
        or image.size == 0
    ):

        return None

    # Make plate larger
    image = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17,
    )

    # Improve image contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        gray
    )

    return enhanced


# ---------------------------------------------------------
# OCR FUNCTION
# ---------------------------------------------------------

def read_number_plate(plate_image):
    """
    Read the number plate text using EasyOCR.

    Returns:

    plate_text
    confidence
    """

    processed = preprocess_plate(
        plate_image
    )

    if processed is None:

        return None, 0.0

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        allowlist=(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
        ),
    )

    best_text = None
    best_confidence = 0.0

    for (
        bounding_box,
        text,
        confidence,
    ) in results:

        cleaned_text = clean_plate_text(
            text
        )

        corrected_text = normalize_indian_plate(
            cleaned_text
        )

        if len(corrected_text) < 6:

            continue

        adjusted_confidence = confidence

        # Give extra preference if the text
        # matches an Indian plate format
        if is_possible_indian_plate(
            corrected_text
        ):

            adjusted_confidence += 0.20

        if (
            adjusted_confidence
            > best_confidence
        ):

            best_text = corrected_text

            best_confidence = (
                adjusted_confidence
            )

    return (
        best_text,
        best_confidence,
    )


# ---------------------------------------------------------
# FIND POSSIBLE NUMBER PLATE REGIONS
# ---------------------------------------------------------

def find_plate_candidates(
    vehicle_image,
):
    """
    Find rectangular regions that
    may contain a number plate.
    """

    if (
        vehicle_image is None
        or vehicle_image.size == 0
    ):

        return []

    height, width = (
        vehicle_image.shape[:2]
    )

    gray = cv2.cvtColor(
        vehicle_image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.bilateralFilter(
        gray,
        11,
        17,
        17,
    )

    edges = cv2.Canny(
        blurred,
        30,
        200,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:50]

    candidates = []

    vehicle_area = (
        width * height
    )

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approx) != 4:

            continue

        x, y, w, h = (
            cv2.boundingRect(
                approx
            )
        )

        if h == 0:

            continue

        aspect_ratio = (
            w / h
        )

        area = (
            w * h
        )

        area_ratio = (
            area / vehicle_area
        )

        # Typical number plate proportions
        if (
            2.0
            <= aspect_ratio
            <= 6.5
            and
            0.003
            <= area_ratio
            <= 0.30
        ):

            plate_crop = (
                vehicle_image[
                    y:y + h,
                    x:x + w,
                ]
            )

            candidates.append(
                (
                    plate_crop,
                    (
                        x,
                        y,
                        w,
                        h,
                    ),
                )
            )

    return candidates


# ---------------------------------------------------------
# DETECT AND READ NUMBER PLATE
# ---------------------------------------------------------

def detect_and_read_plate(
    vehicle_image,
):
    """
    Detect possible number plates
    and read them using OCR.

    Returns:

    plate_text
    confidence
    plate_bbox
    """

    candidates = (
        find_plate_candidates(
            vehicle_image
        )
    )

    best_plate = None

    best_confidence = 0.0

    best_bbox = None

    for (
        plate_crop,
        bbox,
    ) in candidates:

        text, confidence = (
            read_number_plate(
                plate_crop
            )
        )

        if text is None:

            continue

        if (
            confidence
            > best_confidence
        ):

            best_plate = text

            best_confidence = (
                confidence
            )

            best_bbox = bbox

    return (
        best_plate,
        best_confidence,
        best_bbox,
    )


# ---------------------------------------------------------
# TEST NUMBER PLATE READER
# ---------------------------------------------------------

def test_plate_reader():
    """
    Put your test image here:

    data/input/plate_test.jpg

    Then run:

    python src/plate_reader.py
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    image_path = (
        project_root
        / "data"
        / "input"
        / "plate_test.jpg"
    )

    if not image_path.exists():

        print(
            "\nTest image not found."
        )

        print(
            "\nPut the image here:"
        )

        print(
            image_path
        )

        return

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Unable to open image."
        )

        return

    plate_text, confidence, bbox = (
        detect_and_read_plate(
            image
        )
    )

    print(
        "\nDetected Plate:",
        plate_text,
    )

    print(
        "OCR Confidence:",
        round(
            confidence,
            3,
        ),
    )

    if bbox is not None:

        x, y, w, h = bbox

        cv2.rectangle(
            image,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            (
                0,
                255,
                0,
            ),
            3,
        )

        if plate_text:

            cv2.putText(
                image,
                plate_text,
                (
                    x,
                    max(
                        y - 15,
                        30,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

    cv2.imshow(
        "AI Number Plate Recognition",
        image,
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    test_plate_reader()