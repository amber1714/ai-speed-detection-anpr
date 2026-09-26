import cv2


def find_plate_candidates(vehicle_image):
    if vehicle_image is None or vehicle_image.size == 0:
        return []

    height, width = vehicle_image.shape[:2]
    vehicle_area = width * height

    if vehicle_area == 0:
        return []

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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3),
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
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
    )[:80]

    candidates = []

    for contour in contours:
        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(
            approx
        )

        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / h
        plate_area = w * h
        area_ratio = plate_area / vehicle_area

        if (
            1.8 <= aspect_ratio <= 7.0
            and 0.002 <= area_ratio <= 0.30
        ):
            plate_crop = vehicle_image[
                y:y + h,
                x:x + w,
            ]

            if plate_crop.size == 0:
                continue

            candidates.append(
                (
                    plate_crop,
                    (x, y, w, h),
                )
            )

    candidates.sort(
        key=lambda item: (
            item[1][2]
            * item[1][3]
        ),
        reverse=True,
    )

    return candidates