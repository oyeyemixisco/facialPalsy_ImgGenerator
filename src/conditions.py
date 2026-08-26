from config import FACIAL_PALSY_CLASSES, PALSY_CONDITIONS, CLASS_PROMPTS, CLASS_SCORE_THRESHOLDS


def get_all_classes():
    return FACIAL_PALSY_CLASSES


def get_condition(class_name):
    return PALSY_CONDITIONS.get(class_name)


def get_prompt(class_name):
    return CLASS_PROMPTS.get(
        class_name,
        "realistic clinical headshot of one adult patient, frontal view, centered face, plain gray background"
    )


def get_score_threshold(class_name):
    return CLASS_SCORE_THRESHOLDS.get(class_name)



TRIGGER_WORDS = {
    "Healthy": "fpalsy_healthy",
    "Mild_Left_Palsy": "fpalsy_mild_left",
    "Moderate_Left_Palsy": "fpalsy_moderate_left",
    "Severe_Left_Palsy": "fpalsy_severe_left",
    "Mild_Right_Palsy": "fpalsy_mild_right",
    "Moderate_Right_Palsy": "fpalsy_moderate_right",
    "Severe_Right_Palsy": "fpalsy_severe_right"
}

def build_conditioned_prompt(class_name):
    trigger = TRIGGER_WORDS.get(class_name, "")
    base_prompt = get_prompt(class_name)

    if class_name == "Healthy":
        return f"{trigger}, clinical frontal headshot of one adult patient, healthy symmetric face, neutral expression, gray background"

    if class_name == "Mild_Left_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, mild left facial palsy, slight left mouth droop, mild left eyelid weakness, slight facial asymmetry, neutral expression, gray background"

    if class_name == "Moderate_Left_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, moderate left facial palsy, visible left mouth droop, left eyelid partly closed, facial asymmetry, neutral expression, gray background"

    if class_name == "Severe_Left_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, severe left facial palsy, left mouth droop, left eyelid partly closed, left eyebrow lower, facial asymmetry, neutral expression, gray background"

    if class_name == "Mild_Right_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, mild right facial palsy, slight right mouth droop, mild right eyelid weakness, slight facial asymmetry, neutral expression, gray background"

    if class_name == "Moderate_Right_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, moderate right facial palsy, visible right mouth droop, right eyelid partly closed, facial asymmetry, neutral expression, gray background"

    if class_name == "Severe_Right_Palsy":
        return f"{trigger}, clinical frontal headshot of one adult patient, severe right facial palsy, right mouth droop, right eyelid partly closed, right eyebrow lower, facial asymmetry, neutral expression, gray background"

    return f"{trigger}, clinical frontal headshot of one adult patient, neutral expression, gray background"
