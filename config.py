import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KAGGLE_GENERATED_DIR = os.path.join(BASE_DIR, "datasets", "kaggle_generated")

ACCEPTED_DIR = os.path.join(BASE_DIR, "datasets", "accepted")
REJECTED_DIR = os.path.join(BASE_DIR, "datasets", "rejected")
FINAL_DATASET_DIR = os.path.join(BASE_DIR, "datasets", "Final_Synthetic_Facial_Palsy_Database")

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

FACIAL_PALSY_CLASSES = [
    "Healthy",
    "Mild_Palsy",
    "Moderate_Palsy",
    "Severe_Palsy"
]