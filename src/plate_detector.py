from functools import lru_cache

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


MODEL_REPO = "ml-debi/yolov8-license-plate-detection"
MODEL_FILE = "best.onnx"


@lru_cache(maxsize=1)
def get_plate_model():
    """
    Download and load the pretrained license-plate YOLO model.

    The model is downloaded only once and then cached.
    """

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    model = YOLO(
        model_path,
        task="detect",
    )

    return model


def find_plate_candidates(vehicle_image):
    """
    Detect license plates inside a vehicle image.

    Returns:
        [
            (
                plate_crop,
                (x, y, width, height)
            )
        ]
    """

    if (
        vehicle_image is None
        or vehicle_image.size == 0
    ):
        return []

    model = get_plate_model()

    results = model.predict(
        source=vehicle_image,
        conf=0.25,
        imgsz=640,
        verbose=False,
    )

    candidates = []

    if not results:
        return candidates

    result = results[0]

    if result.boxes is None:
        return candidates

    image_height, image_width = vehicle_image.shape[:2]

    detections = []

    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()

        x1, y1, x2, y2 = map(
            int,
            coordinates,
        )

        confidence = float(
            box.conf[0]
        )

        x1 = max(
            0,
            min(x1, image_width - 1),
        )

        y1 = max(
            0,
            min(y1, image_height - 1),
        )

        x2 = max(
            0,
            min(x2, image_width),
        )

        y2 = max(
            0,
            min(y2, image_height),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue

        # Add a small margin around the plate.
        plate_width = x2 - x1
        plate_height = y2 - y1

        padding_x = int(
            plate_width * 0.05
        )

        padding_y = int(
            plate_height * 0.10
        )

        crop_x1 = max(
            0,
            x1 - padding_x,
        )

        crop_y1 = max(
            0,
            y1 - padding_y,
        )

        crop_x2 = min(
            image_width,
            x2 + padding_x,
        )

        crop_y2 = min(
            image_height,
            y2 + padding_y,
        )

        plate_crop = vehicle_image[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if plate_crop.size == 0:
            continue

        width = crop_x2 - crop_x1
        height = crop_y2 - crop_y1

        detections.append(
            (
                confidence,
                plate_crop,
                (
                    crop_x1,
                    crop_y1,
                    width,
                    height,
                ),
            )
        )

    # Highest-confidence detection first.
    detections.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for (
        confidence,
        plate_crop,
        bounding_box,
    ) in detections:

        candidates.append(
            (
                plate_crop,
                bounding_box,
            )
        )

    return candidates