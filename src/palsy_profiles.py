LANDMARK_GROUPS = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,

    "mouth_left_corner": 61,
    "mouth_right_corner": 291,

    # Safer mouth clusters: avoid central/chin point 17
    "mouth_left_cluster": [61, 78, 95, 88, 178],
    "mouth_right_cluster": [291, 308, 324, 318, 402],

    "left_eye_upper": [159, 160, 158],
    "left_eye_lower": [145, 144, 153],

    "right_eye_upper": [386, 387, 385],
    "right_eye_lower": [374, 373, 380],

    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
}

# Severity settings
SEVERITY_SETTINGS = {
    "healthy": {
        "mouth_drop": 0.00,
        "eye_close": 0.00,
        "brow_drop": 0.00,
    },
    "mild": {
        "mouth_drop": 0.04,
        "eye_close": 0.02,
        "brow_drop": 0.015,
    },
    "moderate": {
        "mouth_drop": 0.075,
        "eye_close": 0.035,
        "brow_drop": 0.025,
    },
    "severe": {
        "mouth_drop": 0.11,
        "eye_close": 0.05,
        "brow_drop": 0.035,
    },
}



# Class definitions
FACIAL_PALSY_CLASSES = {
    "Healthy": {
        "side": "none",
        "severity": "healthy",
        "display_name": "Healthy"
    },

    "Mild_Left_Palsy": {
        "side": "left",
        "severity": "mild",
        "display_name": "Mild Left Palsy"
    },
    "Moderate_Left_Palsy": {
        "side": "left",
        "severity": "moderate",
        "display_name": "Moderate Left Palsy"
    },
    "Severe_Left_Palsy": {
        "side": "left",
        "severity": "severe",
        "display_name": "Severe Left Palsy"
    },

    "Mild_Right_Palsy": {
        "side": "right",
        "severity": "mild",
        "display_name": "Mild Right Palsy"
    },
    "Moderate_Right_Palsy": {
        "side": "right",
        "severity": "moderate",
        "display_name": "Moderate Right Palsy"
    },
    "Severe_Right_Palsy": {
        "side": "right",
        "severity": "severe",
        "display_name": "Severe Right Palsy"
    },
}


def get_all_classes():
    return list(FACIAL_PALSY_CLASSES.keys())


def get_class_profile(class_name):
    """
    Returns the merged class profile
    """
    if class_name not in FACIAL_PALSY_CLASSES:
        raise ValueError(f"Unknown class_name: {class_name}")

    base = FACIAL_PALSY_CLASSES[class_name]
    severity_key = base["severity"]
    sev = SEVERITY_SETTINGS[severity_key]

    merged = {
        **base,
        **sev
    }
    return merged


def get_side_groups(side):
    """
    Clinical convention:
    - Patient's RIGHT side appears on viewer's LEFT.
    - Patient's LEFT side appears on viewer's RIGHT.

    MediaPipe/image coordinates are viewer-based here, so we intentionally swap.
    """

    if side == "right":
        # Patient-right = viewer-left
        return {
            "mouth_cluster": LANDMARK_GROUPS["mouth_left_cluster"],
            "mouth_corner": LANDMARK_GROUPS["mouth_left_corner"],
            "eye_upper": LANDMARK_GROUPS["left_eye_upper"],
            "eye_lower": LANDMARK_GROUPS["left_eye_lower"],
            "eyebrow": LANDMARK_GROUPS["left_eyebrow"],
        }

    elif side == "left":
        # Patient-left = viewer-right
        return {
            "mouth_cluster": LANDMARK_GROUPS["mouth_right_cluster"],
            "mouth_corner": LANDMARK_GROUPS["mouth_right_corner"],
            "eye_upper": LANDMARK_GROUPS["right_eye_upper"],
            "eye_lower": LANDMARK_GROUPS["right_eye_lower"],
            "eyebrow": LANDMARK_GROUPS["right_eyebrow"],
        }

    elif side == "none":
        return None

    else:
        raise ValueError(f"Invalid side: {side}")