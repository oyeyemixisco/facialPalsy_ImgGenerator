import os
import json
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def load_image_bgr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def extract_face_landmarks(
    image_bgr,
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
):
    """
    Extracts MediaPipe face landmarks from a BGR image.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return {
            "success": False,
            "image_width": w,
            "image_height": h,
            "landmarks_px": None,
            "landmarks_norm": None,
            "bbox": None,
            "message": "No face detected."
        }

    face_landmarks = results.multi_face_landmarks[0].landmark

    landmarks_px = []
    landmarks_norm = []

    xs = []
    ys = []

    for lm in face_landmarks:
        x_px = lm.x * w
        y_px = lm.y * h

        landmarks_px.append([x_px, y_px])
        landmarks_norm.append({
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z)
        })

        xs.append(x_px)
        ys.append(y_px)

    landmarks_px = np.array(landmarks_px, dtype=np.float32)

    x_min = int(max(min(xs), 0))
    y_min = int(max(min(ys), 0))
    x_max = int(min(max(xs), w - 1))
    y_max = int(min(max(ys), h - 1))

    return {
        "success": True,
        "image_width": w,
        "image_height": h,
        "landmarks_px": landmarks_px,
        "landmarks_norm": landmarks_norm,
        "bbox": [x_min, y_min, x_max, y_max],
        "message": "Face detected successfully."
    }


def extract_face_landmarks_from_path(image_path, **kwargs):
    image_bgr = load_image_bgr(image_path)
    result = extract_face_landmarks(image_bgr, **kwargs)
    result["image_path"] = image_path
    return result


def save_landmark_metadata(result, output_json_path):
    """
    Saves landmark extraction output as JSON.
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    serializable = result.copy()

    if isinstance(serializable.get("landmarks_px"), np.ndarray):
        serializable["landmarks_px"] = serializable["landmarks_px"].tolist()

    with open(output_json_path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_landmark_metadata(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if data.get("landmarks_px") is not None:
        data["landmarks_px"] = np.array(data["landmarks_px"], dtype=np.float32)

    return data


def draw_landmark_overlay(
    image_bgr,
    landmarks_px,
    output_path=None,
    selected_indices=None,
    all_points_color=(0, 255, 0),
    selected_color=(0, 0, 255),
    radius=1
):
    
    # Draws landmarks on an image

    overlay = image_bgr.copy()

    for i, (x, y) in enumerate(landmarks_px):
        if selected_indices and i in selected_indices:
            color = selected_color
            point_radius = max(radius + 1, 2)
        else:
            color = all_points_color
            point_radius = radius

        cv2.circle(overlay, (int(x), int(y)), point_radius, color, -1)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)

    return overlay


def get_landmark_indices_summary():
    """
    Utility helper for debugging or documentation.
    """
    return {
        "total_landmarks_expected": "468 or 478 depending on MediaPipe/refine_landmarks version"
    }