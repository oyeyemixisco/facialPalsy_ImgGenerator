import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

import uuid

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# -----------------------------
# MediaPipe landmark indices
# -----------------------------

MOUTH_CORNER_A = 61
MOUTH_CORNER_B = 291

# Eye landmark groups
EYE_GROUP_A = {
    "outer": 33,
    "inner": 133,
    "top": 159,
    "bottom": 145
}

EYE_GROUP_B = {
    "outer": 263,
    "inner": 362,
    "top": 386,
    "bottom": 374
}

# Brow landmark candidates
BROW_A = 70
BROW_B = 300

# Face scale landmarks
FACE_LEFT = 234
FACE_RIGHT = 454

# Extra lower-face landmarks for structural asymmetry
NOSE_TIP = 1
CHIN = 152

UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LEFT_CHEEK = 50
RIGHT_CHEEK = 280

LEFT_NASOLABIAL = 205
RIGHT_NASOLABIAL = 425


# -----------------------------
# New calibrated severity ranges
# -----------------------------
# These scores are now scaled between 0 and 1.
# They are more realistic for weighted landmark-based validation.

CLASS_SCORE_THRESHOLDS = {
    "Healthy": (0.00, 0.13),
    "Mild_Palsy": (0.13, 0.23),
    "Moderate_Palsy": (0.23, 0.34),
    "Severe_Palsy": (0.34, 1.00),
}


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_bgr


def extract_landmarks(image_bgr):
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        points = []
        for lm in landmarks:
            points.append({
                "x": float(lm.x * w),
                "y": float(lm.y * h),
                "z": float(lm.z)
            })

        return points


def p(points, idx):
    return np.array([points[idx]["x"], points[idx]["y"]], dtype=np.float32)


def distance(a, b):
    return float(np.linalg.norm(a - b))


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def get_face_width(points):
    left = p(points, FACE_LEFT)
    right = p(points, FACE_RIGHT)
    width = distance(left, right)

    if width <= 0:
        return 1.0

    return width


def get_viewer_left_right_points(point_a, point_b):
    """
    In image coordinates:
    smaller x = viewer's left
    larger x = viewer's right
    """
    if point_a[0] <= point_b[0]:
        return point_a, point_b

    return point_b, point_a


def viewer_side_to_patient_side(viewer_side):
    if viewer_side == "viewer_left":
        return "right"

    if viewer_side == "viewer_right":
        return "left"

    return "none"


def get_expected_side(class_name):
    return "not_applicable (future work)"


def get_expected_range(class_name):
    return CLASS_SCORE_THRESHOLDS.get(class_name, (0.00, 1.00))

def classify_severity_from_score(score):
    """
    Convert composite palsy score into predicted severity label.
    """
    if score < 0.13:
        return "Healthy"
    elif score < 0.23:
        return "Mild"
    elif score < 0.34:
        return "Moderate"
    else:
        return "Severe"


def get_eye_measurements(points):
    """
    Returns eye opening for viewer-left and viewer-right eyes.
    This avoids relying on MediaPipe landmark names as clinical side labels.
    """

    eye_a_center = (
        p(points, EYE_GROUP_A["outer"]) +
        p(points, EYE_GROUP_A["inner"])
    ) / 2.0

    eye_b_center = (
        p(points, EYE_GROUP_B["outer"]) +
        p(points, EYE_GROUP_B["inner"])
    ) / 2.0

    eye_a_opening = distance(
        p(points, EYE_GROUP_A["top"]),
        p(points, EYE_GROUP_A["bottom"])
    )

    eye_b_opening = distance(
        p(points, EYE_GROUP_B["top"]),
        p(points, EYE_GROUP_B["bottom"])
    )

    if eye_a_center[0] <= eye_b_center[0]:
        return {
            "viewer_left_opening": eye_a_opening,
            "viewer_right_opening": eye_b_opening,
            "viewer_left_center": eye_a_center,
            "viewer_right_center": eye_b_center
        }

    return {
        "viewer_left_opening": eye_b_opening,
        "viewer_right_opening": eye_a_opening,
        "viewer_left_center": eye_b_center,
        "viewer_right_center": eye_a_center
    }


def compute_mouth_features(points):
    face_width = get_face_width(points)

    mouth_a = p(points, MOUTH_CORNER_A)
    mouth_b = p(points, MOUTH_CORNER_B)

    viewer_left_mouth, viewer_right_mouth = get_viewer_left_right_points(
        mouth_a,
        mouth_b
    )

    vertical_difference = abs(viewer_left_mouth[1] - viewer_right_mouth[1])
    raw_mouth_asymmetry = vertical_difference / face_width

    # Convert raw value into stronger severity score.
    # A raw value around 0.16+ becomes very strong.
    mouth_score = clamp(raw_mouth_asymmetry / 0.16)

    if viewer_left_mouth[1] > viewer_right_mouth[1]:
        affected_viewer_side = "viewer_left"
    elif viewer_right_mouth[1] > viewer_left_mouth[1]:
        affected_viewer_side = "viewer_right"
    else:
        affected_viewer_side = "none"

    affected_patient_side = viewer_side_to_patient_side(affected_viewer_side)

    return {
        "raw": raw_mouth_asymmetry,
        "score": mouth_score,
        "affected_patient_side": affected_patient_side
    }


def compute_eye_features(points):
    measurements = get_eye_measurements(points)

    left_open = measurements["viewer_left_opening"]
    right_open = measurements["viewer_right_opening"]

    max_open = max(left_open, right_open, 1e-6)
    min_open = min(left_open, right_open)

    # If one eye is much more closed than the other, this becomes high.
    closure_difference_ratio = (max_open - min_open) / max_open
    eye_score = clamp(closure_difference_ratio)

    if left_open < right_open:
        affected_viewer_side = "viewer_left"
    elif right_open < left_open:
        affected_viewer_side = "viewer_right"
    else:
        affected_viewer_side = "none"

    affected_patient_side = viewer_side_to_patient_side(affected_viewer_side)

    return {
        "raw": closure_difference_ratio,
        "score": eye_score,
        "affected_patient_side": affected_patient_side
    }


def compute_brow_features(points):
    face_width = get_face_width(points)

    brow_a = p(points, BROW_A)
    brow_b = p(points, BROW_B)

    viewer_left_brow, viewer_right_brow = get_viewer_left_right_points(
        brow_a,
        brow_b
    )

    vertical_difference = abs(viewer_left_brow[1] - viewer_right_brow[1])
    raw_brow_asymmetry = vertical_difference / face_width

    # Brow changes are usually smaller, so scale more gently.
    brow_score = clamp(raw_brow_asymmetry / 0.10)

    if viewer_left_brow[1] > viewer_right_brow[1]:
        affected_viewer_side = "viewer_left"
    elif viewer_right_brow[1] > viewer_left_brow[1]:
        affected_viewer_side = "viewer_right"
    else:
        affected_viewer_side = "none"

    affected_patient_side = viewer_side_to_patient_side(affected_viewer_side)

    return {
        "raw": raw_brow_asymmetry,
        "score": brow_score,
        "affected_patient_side": affected_patient_side
    }


def compute_global_symmetry_error(points):
    face_width = get_face_width(points)

    bilateral_pairs = [
        (61, 291),    # mouth corners
        (33, 263),    # eye outer areas
        (133, 362),   # eye inner areas
        (70, 300),    # brows
        (234, 454),   # cheeks
        (93, 323),    # lower cheeks
    ]

    raw_values = []

    for idx_a, idx_b in bilateral_pairs:
        pa = p(points, idx_a)
        pb = p(points, idx_b)

        vertical_difference = abs(pa[1] - pb[1])
        raw_values.append(vertical_difference / face_width)

    raw_global = float(np.mean(raw_values))

    # Scale global asymmetry into 0-1 range.
    global_score = clamp(raw_global / 0.15)

    return {
        "raw": raw_global,
        "score": global_score
    }


def compute_lower_face_features(points):
    """
    Detects lower-face structural asymmetry that may not be captured by
    simple mouth-corner vertical difference.
    """

    face_width = get_face_width(points)

    if face_width <= 0:
        return {
            "score": 0.0,
            "mouth_center_deviation": 0.0,
            "lip_chin_deviation": 0.0,
            "cheek_asymmetry": 0.0,
            "nasolabial_asymmetry": 0.0,
        }

    # Convert dictionary landmarks into [x, y] arrays
    nose = p(points, NOSE_TIP)
    chin = p(points, CHIN)

    # Face midline using nose and chin
    mid_x = (nose[0] + chin[0]) / 2.0

    # Mouth centre based on both mouth corners
    mouth_left = p(points, MOUTH_LEFT)
    mouth_right = p(points, MOUTH_RIGHT)
    mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2.0

    mouth_center_deviation = abs(mouth_center_x - mid_x) / face_width

    # Lip centre deviation from the face midline
    upper_lip = p(points, UPPER_LIP_CENTER)
    lower_lip = p(points, LOWER_LIP_CENTER)
    lip_center_x = (upper_lip[0] + lower_lip[0]) / 2.0

    lip_chin_deviation = abs(lip_center_x - mid_x) / face_width

    # Cheek/nasolabial asymmetry using x-distance from nose
    left_cheek = p(points, LEFT_CHEEK)
    right_cheek = p(points, RIGHT_CHEEK)

    left_cheek_dist = abs(left_cheek[0] - nose[0])
    right_cheek_dist = abs(right_cheek[0] - nose[0])
    cheek_asymmetry = abs(left_cheek_dist - right_cheek_dist) / face_width

    left_naso = p(points, LEFT_NASOLABIAL)
    right_naso = p(points, RIGHT_NASOLABIAL)

    left_naso_dist = abs(left_naso[0] - nose[0])
    right_naso_dist = abs(right_naso[0] - nose[0])
    nasolabial_asymmetry = abs(left_naso_dist - right_naso_dist) / face_width

    raw_score = (
        0.35 * mouth_center_deviation +
        0.30 * lip_chin_deviation +
        0.20 * cheek_asymmetry +
        0.15 * nasolabial_asymmetry
    )

    # Keep this conservative. Lower-face should support the decision,
    # not dominate the whole score.
    score = clamp(raw_score / 0.16)

    return {
        "score": score,
        "mouth_center_deviation": mouth_center_deviation,
        "lip_chin_deviation": lip_chin_deviation,
        "cheek_asymmetry": cheek_asymmetry,
        "nasolabial_asymmetry": nasolabial_asymmetry,
    }


def choose_detected_side(mouth_features, eye_features, brow_features):
    """
    Decide affected patient side.

    Improved strategy:
    1. Strong eye asymmetry is treated as a highly reliable side cue.
    2. Mouth asymmetry is useful, but it can point toward the pulled/stronger side,
       so it should not always dominate.
    3. If eye and mouth strongly disagree, prefer eye for severe-looking cases.
    4. For subtle cases, use weighted voting.
    """

    mouth_side = mouth_features["affected_patient_side"]
    eye_side = eye_features["affected_patient_side"]
    brow_side = brow_features["affected_patient_side"]

    mouth_score = mouth_features["score"]
    eye_score = eye_features["score"]
    brow_score = brow_features["score"]

    # -------------------------------------------------
    # Rule 1: strong eye closure asymmetry should dominate.
    # This fixes cases where mouth/brow point right but the affected eye points left.
    # -------------------------------------------------
    if eye_score >= 0.25 and eye_side in ["left", "right"]:
        # If mouth disagrees but is not much stronger than eye, trust the eye.
        if mouth_side != eye_side:
            if eye_score >= (mouth_score - 0.08):
                confidence = eye_score
                return eye_side, confidence

        # If mouth agrees with eye, even stronger confidence.
        if mouth_side == eye_side:
            confidence = min(1.0, eye_score + (0.25 * mouth_score))
            return eye_side, confidence

        # If brow agrees with eye, support the eye side.
        if brow_side == eye_side:
            confidence = min(1.0, eye_score + (0.15 * brow_score))
            return eye_side, confidence

    # -------------------------------------------------
    # Rule 2: if mouth is very strong and eye is weak, use mouth.
    # This handles cases where eye asymmetry is not visible but mouth droop is clear.
    # -------------------------------------------------
    if mouth_score >= 0.35 and eye_score < 0.18 and mouth_side in ["left", "right"]:
        return mouth_side, mouth_score

    # -------------------------------------------------
    # Rule 3: if mouth is moderate and eye/brow do not strongly disagree,
    # trust mouth.
    # -------------------------------------------------
    if mouth_score >= 0.22 and mouth_side in ["left", "right"]:
        agreeing_evidence = 0.0
        opposing_evidence = 0.0

        if eye_side == mouth_side:
            agreeing_evidence += eye_score
        elif eye_side in ["left", "right"]:
            opposing_evidence += eye_score

        if brow_side == mouth_side:
            agreeing_evidence += brow_score
        elif brow_side in ["left", "right"]:
            opposing_evidence += brow_score

        if opposing_evidence <= agreeing_evidence + 0.10:
            confidence = mouth_score + (0.20 * agreeing_evidence) - (0.20 * opposing_evidence)
            return mouth_side, max(0.0, min(1.0, confidence))

    # -------------------------------------------------
    # Rule 4: weighted vote for subtle or unclear cases.
    # Eye and mouth are weighted more than brow.
    # -------------------------------------------------
    side_scores = {
        "left": 0.0,
        "right": 0.0,
        "none": 0.0
    }

    side_scores[mouth_side] += 0.40 * mouth_score
    side_scores[eye_side] += 0.45 * eye_score
    side_scores[brow_side] += 0.15 * brow_score

    left_score = side_scores["left"]
    right_score = side_scores["right"]

    strongest_score = max(left_score, right_score)
    confidence_gap = abs(left_score - right_score)

    if strongest_score < 0.08:
        return "none", 0.0

    if confidence_gap < 0.06:
        return "uncertain", confidence_gap

    if left_score > right_score:
        return "left", confidence_gap

    return "right", confidence_gap

def compute_composite_score(mouth_features, eye_features, brow_features, global_features,
    lower_face_features=None, return_details=False):

    mouth_score = float(mouth_features["score"])
    eye_score = float(eye_features["score"])
    brow_score = float(brow_features["score"])
    global_score = float(global_features["score"])

    lower_face_score = 0.0

    if lower_face_features is not None:
        lower_face_score = float(lower_face_features.get("score", 0.0))

    # Raw composite before correction
    raw_composite = (
        0.34 * mouth_score +
        0.34 * eye_score +
        0.08 * brow_score +
        0.10 * global_score +
        0.14 * lower_face_score
    )

    active_expression = False
    expression_reason = ""
    quality_note = ""
    artifact_type = "none"

    corrected_mouth = mouth_score
    corrected_eye = eye_score
    corrected_brow = brow_score
    corrected_global = global_score
    corrected_lower_face = lower_face_score

    # Eyebrow false-positive protection
    if mouth_score < 0.16 and eye_score < 0.18 and brow_score > 0.35:
        corrected_brow = 0.18
        active_expression = False
        expression_reason = "Eyebrow landmark distortion detected"
        artifact_type = "eyebrow_distortion"
        quality_note = expression_reason

    # Active wink / synkinesis-like expression filter
    if eye_score > 0.50 and brow_score < 0.22:

        # Case A: eye is very high and mouth/global also show asymmetry.
        # This is likely palsy-related dynamic asymmetry, not a simple wink.
        if mouth_score >= 0.24 or global_score >= 0.16:
            active_expression = False
            expression_reason = "Palsy-related dynamic eye-mouth asymmetry detected"
            artifact_type = "dynamic_palsy_pattern"
            quality_note = expression_reason

            # Reduce eye slightly so it does not force Grade V/VI alone,
            # but keep enough strength to support Grade IV.
            corrected_eye = min(eye_score, 0.42)

            # Keep mouth contribution because it supports palsy pattern.
            corrected_mouth = mouth_score

        # Case B: eye is high but mouth/global are low.
        # This is more likely a simple wink or non-neutral eye closure.
        else:
            active_expression = True
            expression_reason = "Possible wink or non-neutral eye closure detected"
            quality_note = expression_reason
            artifact_type = "possible_wink"

            corrected_eye = min(eye_score, 0.34)

    # Head tilt / perspective distortion filter
    if (
        mouth_score > 0.38 and
        brow_score > 0.35 and
        global_score > 0.30 and
        eye_score < 0.14
    ):
        active_expression = False
        expression_reason = "Possible head tilt or perspective distortion detected"
        quality_note = expression_reason
        artifact_type = "pose_distortion"

        corrected_mouth = min(mouth_score, 0.18)
        corrected_brow = min(brow_score, 0.18)
        corrected_global = min(global_score, 0.18)
        corrected_lower_face = min(lower_face_score, 0.18)


    # Active smile / mouth contraction filter
    if (
        mouth_score > 0.40 and
        eye_score < 0.18 and
        brow_score < 0.22 and
        global_score < 0.22
    ):
        active_expression = True
        expression_reason = "Active mouth movement or smile detected"
        quality_note = expression_reason
        artifact_type = "mouth_expression"

        corrected_mouth = min(mouth_score, 0.25)

    # Soft caps to avoid one landmark region dominating
    corrected_mouth = min(corrected_mouth, 0.50)
    corrected_eye = min(corrected_eye, 0.70)
    corrected_brow = min(corrected_brow, 0.35)
    corrected_global = min(corrected_global, 0.35)
    corrected_lower_face = min(corrected_lower_face, 0.35)

    if (
        corrected_mouth < 0.13 and
        corrected_eye < 0.13 and
        corrected_brow < 0.18 and
        lower_face_score >= 0.28
    ):
        artifact_type = "lower_face_structural_asymmetry"
        quality_note = "Lower-face structural asymmetry detected despite low standard composite score"
        expression_reason = quality_note

        # Lift the corrected mouth slightly so the final grade can move from
        # Grade I to Grade II where appropriate.
        corrected_mouth = max(corrected_mouth, 0.18)


    corrected_composite = (
        0.34 * corrected_mouth +
        0.34 * corrected_eye +
        0.08 * corrected_brow +
        0.10 * corrected_global +
        0.14 * corrected_lower_face
    )

    corrected_composite = clamp(corrected_composite)
    raw_composite = clamp(raw_composite)

    if return_details:
        return {
            "raw_score": raw_composite,
            "corrected_score": corrected_composite,
            "active_expression": active_expression,
            "expression_reason": expression_reason,
            "quality_note": quality_note,
            "artifact_type": artifact_type,
            "corrected_mouth": corrected_mouth,
            "corrected_eye": corrected_eye,
            "corrected_brow": corrected_brow,
            "corrected_global": corrected_global,
            "lower_face_score": lower_face_score,
            "corrected_lower_face": corrected_lower_face,
        }

    return corrected_composite


def classify_grade_from_features(
    composite_score,
    mouth_score,
    eye_score,
    brow_score,
    global_score,
    lower_face_score=0.0,
    artifact_type="none"
):
    """
    Classify final grade using composite score plus local feature checks.

    The lower-face correction is conservative:
    it can only upgrade a false-normal case from Grade I to Grade II.
    It cannot force Moderate, Severe, Grade IV, Grade V or Grade VI.
    """

    if composite_score < 0.13:
        if (
            lower_face_score >= 0.14 and
            mouth_score >= 0.07 and
            global_score >= 0.05 and
            eye_score < 0.16 and
            brow_score < 0.25 and
            artifact_type not in ["pose_distortion", "possible_wink", "mouth_expression"]
        ):
            return "Grade_II_Mild"

    # Grade I / Grade II boundary zone
    if 0.09 <= composite_score < 0.13:

        if mouth_score >= 0.12 or eye_score >= 0.13:
            return "Grade_II_Mild"

        if brow_score >= 0.20 and (mouth_score >= 0.10 or eye_score >= 0.11):
            return "Grade_II_Mild"

        return "Grade_I_Normal"

    # Standard grading
    if composite_score < 0.13:
        return "Grade_I_Normal"

    elif composite_score < 0.18:
        return "Grade_II_Mild"

    elif composite_score < 0.25:
        return "Grade_III_Moderate"

    elif composite_score < 0.36:
        return "Grade_IV_Moderate_Severe"

    elif composite_score < 0.55:
        return "Grade_V_Severe"

    else:
        return "Grade_VI_Total_Paralysis" 


def grade_to_broad_severity(predicted_grade):
    mapping = {
        "Grade_I_Normal": "Healthy",
        "Grade_II_Mild": "Mild",
        "Grade_III_Moderate": "Moderate",
        "Grade_IV_Moderate_Severe": "Moderate-Severe",
        "Grade_V_Severe": "Severe",
        "Grade_VI_Total_Paralysis": "Total Paralysis",
    }

    return mapping.get(predicted_grade, "Unknown")


def classify_grade_from_score(score):
    if score < 0.13:
        return "Grade_I_Normal"
    elif score < 0.18:
        return "Grade_II_Mild"
    elif score < 0.25:
        return "Grade_III_Moderate"
    elif score < 0.36:
        return "Grade_IV_Moderate_Severe"
    elif score < 0.55:
        return "Grade_V_Severe"
    else:
        return "Grade_VI_Total_Paralysis"
      

def grade_to_display_name(grade):
    display_names = {
        "Grade_I_Normal": "Grade I - Normal",
        "Grade_II_Mild": "Grade II - Mild",
        "Grade_III_Moderate": "Grade III - Moderate",
        "Grade_IV_Moderate_Severe": "Grade IV - Moderate-Severe",
        "Grade_V_Severe": "Grade V - Severe",
        "Grade_VI_Total_Paralysis": "Grade VI - Total Paralysis",
    }

    return display_names.get(grade, grade)


def get_expected_severity(class_name):
    
    if class_name == "Healthy":
        return "Healthy"

    if "Mild_Palsy" in class_name:
        return "Mild"

    if "Moderate_Palsy" in class_name:
        return "Moderate"

    if "Severe_Palsy" in class_name:
        return "Severe"

    return "Unknown"


def make_validation_decision(
    class_name,
    composite_score,
    predicted_grade=None,
    detected_side=None,
    side_confidence=0.0,
    active_expression=False,
    expression_reason="",
    quality_note="",
    artifact_type="none"
):
    expected_severity = get_expected_severity(class_name)

    if predicted_grade is None:
        predicted_grade = classify_grade_from_score(composite_score)

    predicted_grade_display = grade_to_display_name(predicted_grade)

    acceptable_grades = expected_grades_for_original_class(class_name)


    print("DEBUG VALIDATION DECISION")
    print("Class name:", class_name)
    print("Composite score:", composite_score)
    print("Predicted grade:", predicted_grade)
    print("Predicted grade display:", predicted_grade_display)
    print("Acceptable grades:", acceptable_grades)
    print("Artifact type:", artifact_type)
    print("Quality note:", quality_note)

    if artifact_type == "dynamic_palsy_pattern":
        if predicted_grade in acceptable_grades:
            return f"Generated Image Accepted - {predicted_grade_display} ({quality_note})"
        return f"Needs Review - {quality_note}; metrics suggest {predicted_grade_display}"
    
    if artifact_type == "lower_face_structural_asymmetry":
        if predicted_grade in acceptable_grades:
            return f"Generated Image Accepted - {predicted_grade_display} ({quality_note})"

        return f"Needs Review - {quality_note}; metrics suggest {predicted_grade_display}"

    # True artefacts that should remain Needs Review
    if artifact_type in ["pose_distortion", "mouth_expression"]:
        if predicted_grade in acceptable_grades:
            return f"Generated Image Accepted - {predicted_grade_display} ({quality_note})"
        return f"Needs Review - {quality_note}; metrics suggest {predicted_grade_display}"

    # Possible wink is more uncertain, so keep it for review unless there is
    # strong eye-mouth/global evidence handled as dynamic_palsy_pattern.
    if artifact_type == "possible_wink":
        return f"Needs Review - {quality_note}; metrics suggest {predicted_grade_display}"
    
    if active_expression:
        reason = expression_reason or quality_note or "Active/non-neutral expression detected"
        return f"Needs Review - {reason}; metrics suggest {predicted_grade_display}"

    if predicted_grade in acceptable_grades:
        return f"Generated Image Accepted - {predicted_grade_display}"

    if class_name == "Healthy" and predicted_grade == "Grade_II_Mild":
        return "Needs Review - Generation mismatch: Healthy image shows mild asymmetry"

    if class_name == "Mild_Palsy" and predicted_grade == "Grade_I_Normal":
        return "Generated Image Accepted - Mild_Palsy sample appears near-normal, classified as Grade I - Normal"

    if class_name == "Mild_Palsy" and predicted_grade == "Grade_III_Moderate":
        return "Needs Review - Generation mismatch: Mild palsy image appears closer to Grade III Moderate"

    if class_name == "Moderate_Palsy" and predicted_grade == "Grade_II_Mild":
        return "Needs Review - Generation mismatch: Moderate palsy image appears closer to Grade II Mild"

    if class_name == "Moderate_Palsy" and predicted_grade == "Grade_V_Severe":
        return "Needs Review - Generation mismatch: Moderate palsy image appears closer to Grade V Severe"

    if class_name == "Severe_Palsy" and predicted_grade == "Grade_III_Moderate":
        return "Needs Review - Generation mismatch: Severe palsy image appears closer to Grade III Moderate"
    
    if class_name == "Severe_Palsy" and predicted_grade == "Grade_I_Normal":
        return (
            "Needs Review - Intended Severe_Palsy, but neutral-frame metrics suggest "
            f"{predicted_grade_display}; expression frame required"
        )

    if class_name == "Severe_Palsy" and predicted_grade == "Grade_II_Mild":
        return (
            "Needs Review - Generation mismatch: intended Severe_Palsy, "
            f"but Validation analysis suggests {predicted_grade_display}"
        )

    return (
        f"Needs Review - Generation mismatch: intended {class_name}, "
        f"metrics suggest {predicted_grade_display}"
    )

def analyse_facial_palsy_image(image_path, class_name):
    image_bgr = load_image(image_path)
    h, w, _ = image_bgr.shape

    points = extract_landmarks(image_bgr)

    if points is None:
        return {
            "success": False,
            "face_detected": "No",
            "landmarks": "0/468",
            "resolution": f"{w} × {h}",
            "message": "No face detected."
        }

    mouth_features = compute_mouth_features(points)
    eye_features = compute_eye_features(points)
    brow_features = compute_brow_features(points)
    global_features = compute_global_symmetry_error(points)
    lower_face_features = compute_lower_face_features(points)

    composite_details = compute_composite_score(
        mouth_features,
        eye_features,
        brow_features,
        global_features,
        lower_face_features=lower_face_features,
        return_details=True
    )

    raw_composite_score = composite_details["raw_score"]
    composite_score = composite_details["corrected_score"]
    active_expression = composite_details["active_expression"]
    expression_reason = composite_details["expression_reason"]
    quality_note = composite_details.get("quality_note", "")
    artifact_type = composite_details.get("artifact_type", "none")

    predicted_grade = classify_grade_from_features(
        composite_score=composite_score,
        mouth_score=mouth_features["score"],
        eye_score=eye_features["score"],
        brow_score=brow_features["score"],
        global_score=global_features["score"],
        lower_face_score=lower_face_features["score"],
        artifact_type=artifact_type
    )

    predicted_grade_display = grade_to_display_name(predicted_grade)


    detected_side, side_confidence = choose_detected_side(
        mouth_features,
        eye_features,
        brow_features
    )

    expected_min, expected_max = get_expected_range(class_name)
    expected_side = get_expected_side(class_name)

    decision = make_validation_decision(
        class_name=class_name,
        composite_score=composite_score,
        predicted_grade=predicted_grade,
        detected_side=detected_side,
        side_confidence=side_confidence,
        active_expression=active_expression,
        expression_reason=expression_reason,
        quality_note=quality_note,
        artifact_type=artifact_type
    )

    overlay_url = create_mesh_overlay(image_path)

    print("DEBUG COMPOSITE CHECK")
    print("Raw composite:", raw_composite_score)
    print("Corrected composite:", composite_score)
    print("Active expression:", active_expression)
    print("Expression reason:", expression_reason)
    print("Predicted grade:", predicted_grade_display)

    return {
        "success": True,
        "face_detected": "Yes",
        "landmarks": "468/468",
        "resolution": f"{w} × {h}",

        # These are now scaled severity scores, not only raw pixel differences.
        "mouth_asymmetry": round(mouth_features["score"], 4),
        "eye_asymmetry": round(eye_features["score"], 4),
        "eyebrow_asymmetry": round(brow_features["score"], 4),
        "global_symmetry_error": round(global_features["score"], 4),

        # Corrected score used for final grade classification
        "composite_palsy_score": round(composite_score, 4),
        "composite_score": round(composite_score, 4),


        "lower_face_asymmetry": round(lower_face_features["score"], 4),
        "mouth_center_deviation": round(lower_face_features["mouth_center_deviation"], 4),
        "lip_chin_deviation": round(lower_face_features["lip_chin_deviation"], 4),
        "cheek_asymmetry": round(lower_face_features["cheek_asymmetry"], 4),
        "nasolabial_asymmetry": round(lower_face_features["nasolabial_asymmetry"], 4),


        # Raw score before active-expression correction
        "raw_composite_score": round(raw_composite_score, 4),

        # Expression / wink / synkinesis-like detection
        "active_expression": active_expression,
        "expression_reason": expression_reason,
        "quality_note": quality_note,
        "artifact_type": artifact_type,

        # Useful extra diagnostic values
        "raw_mouth_asymmetry": round(mouth_features["raw"], 4),
        "raw_eye_asymmetry": round(eye_features["raw"], 4),
        "raw_eyebrow_asymmetry": round(brow_features["raw"], 4),
        "raw_global_symmetry_error": round(global_features["raw"], 4),

        "detected_side": detected_side,
        "expected_side": expected_side,
        "side_confidence": round(side_confidence, 4),

        "predicted_severity": grade_to_broad_severity(predicted_grade),
        "predicted_grade": predicted_grade,
        "predicted_grade_display": predicted_grade_display,

        "expected_range": f"{expected_min:.2f} - {expected_max:.2f}",
        "decision": decision,

        "overlay_image_url": overlay_url
    }

def expected_grades_for_original_class(class_name):
    mapping = {
        "Healthy": [
            "Grade_I_Normal"
        ],

        "Mild_Palsy": [
            "Grade_I_Normal",
            "Grade_II_Mild"
        ],

        "Moderate_Palsy": [
            "Grade_III_Moderate",
            "Grade_IV_Moderate_Severe"
        ],

        "Severe_Palsy": [
            "Grade_IV_Moderate_Severe",
            "Grade_V_Severe",
            "Grade_VI_Total_Paralysis"
        ],
    }

    return mapping.get(class_name, [])


def create_mesh_overlay(image_path, output_dir="static/validation_overlays"):
    """
    Creates a cleaner validation overlay:
    - light landmark dots
    - selected facial feature lines
    - centre vertical guide line
    - horizontal guide line
    This avoids the heavy full-face mesh mask.
    """
    import uuid

    os.makedirs(output_dir, exist_ok=True)

    image_bgr = load_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w, _ = image_bgr.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        annotated_image = image_bgr.copy()

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        def to_pixel(idx):
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        def draw_points(indices, color=(0, 255, 120), radius=2):
            for idx in indices:
                x, y = to_pixel(idx)
                cv2.circle(
                    annotated_image,
                    (x, y),
                    radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA
                )

        def draw_polyline(indices, color=(0, 255, 120), thickness=1, closed=False):
            pts = np.array([to_pixel(idx) for idx in indices], np.int32)

            if len(pts) > 1:
                cv2.polylines(
                    annotated_image,
                    [pts],
                    closed,
                    color,
                    thickness,
                    lineType=cv2.LINE_AA
                )

        # Key landmark groups
        left_eye = [33, 160, 158, 133, 153, 144, 145, 33]
        right_eye = [362, 385, 387, 263, 373, 380, 374, 362]

        left_eyebrow = [70, 63, 105, 66, 107]
        right_eyebrow = [336, 296, 334, 293, 300]

        outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                      375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

        inner_lips = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

        nose_bridge = [168, 6, 197, 195, 5, 4, 1]
        nose_base = [98, 97, 2, 326, 327]

        jawline = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                   152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

        cheeks = [50, 101, 205, 425, 330, 280]

        # Draw subtle guide lines
        cv2.line(
            annotated_image,
            (w // 2, 0),
            (w // 2, h),
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA
        )

        cv2.line(
            annotated_image,
            (0, h // 2),
            (w, h // 2),
            (255, 170, 0),
            1,
            lineType=cv2.LINE_AA
        )

        # Draw facial feature outlines
        draw_polyline(left_eye, color=(0, 255, 120), thickness=2, closed=True)
        draw_polyline(right_eye, color=(0, 255, 120), thickness=2, closed=True)

        draw_polyline(left_eyebrow, color=(0, 255, 120), thickness=2)
        draw_polyline(right_eyebrow, color=(0, 255, 120), thickness=2)

        draw_polyline(outer_lips, color=(255, 255, 255), thickness=2, closed=True)
        draw_polyline(inner_lips, color=(255, 255, 255), thickness=1, closed=True)

        draw_polyline(nose_bridge, color=(0, 220, 255), thickness=1)
        draw_polyline(nose_base, color=(0, 220, 255), thickness=1)

        draw_polyline(jawline, color=(180, 220, 255), thickness=1)

        # Draw landmark dots only on important regions
        key_points = (
            left_eye +
            right_eye +
            left_eyebrow +
            right_eyebrow +
            outer_lips +
            inner_lips +
            nose_bridge +
            nose_base +
            jawline +
            cheeks
        )

        draw_points(key_points, color=(80, 255, 120), radius=2)

        filename = f"overlay_{uuid.uuid4().hex[:10]}.png"
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, annotated_image)

        return f"/static/validation_overlays/{filename}"
