import os
import shutil
import cv2
import mediapipe as mp

REFERENCE_DIR = "static/reference_faces"
PROCESSED_DIR = "static/processed_reference_faces"

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def is_valid_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def check_reference_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "valid": False,
            "reason": "Image could not be read."
        }

    h, w, _ = image.shape

    if w != 512 or h != 512:
        return {
            "valid": False,
            "reason": f"Image size is {w}x{h}, expected 512x512."
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Face detection check
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:

        results = face_detection.process(rgb)
        detections = results.detections if results.detections else []

        if len(detections) == 0:
            return {
                "valid": False,
                "reason": "No face detected."
            }

        if len(detections) > 1:
            return {
                "valid": False,
                "reason": "Multiple faces detected."
            }

        detection = detections[0]
        bbox = detection.location_data.relative_bounding_box

        face_width_ratio = bbox.width
        face_height_ratio = bbox.height

        if face_width_ratio < 0.25 or face_height_ratio < 0.25:
            return {
                "valid": False,
                "reason": "Face too small."
            }

        face_center_x = bbox.xmin + bbox.width / 2
        face_center_y = bbox.ymin + bbox.height / 2

        if abs(face_center_x - 0.5) > 0.25 or abs(face_center_y - 0.5) > 0.25:
            return {
                "valid": False,
                "reason": "Face not centred."
            }

    # 2. Landmark extraction check
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        mesh_results = face_mesh.process(rgb)

        if not mesh_results.multi_face_landmarks:
            return {
                "valid": False,
                "reason": "Facial landmarks not detected."
            }

        landmark_count = len(mesh_results.multi_face_landmarks[0].landmark)

        if landmark_count < 400:
            return {
                "valid": False,
                "reason": f"Insufficient landmarks detected: {landmark_count}."
            }

    return {
        "valid": True,
        "reason": "Reference image passed quality checks."
    }


def prepare_reference_faces():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(REFERENCE_DIR)
        if is_valid_image_file(f)
    ]

    report = []

    for filename in files:
        source_path = os.path.join(REFERENCE_DIR, filename)
        result = check_reference_face(source_path)

        if result["valid"]:
            target_path = os.path.join(PROCESSED_DIR, filename)
            shutil.copy2(source_path, target_path)

        report.append({
            "filename": filename,
            "valid": result["valid"],
            "reason": result["reason"]
        })

    return report