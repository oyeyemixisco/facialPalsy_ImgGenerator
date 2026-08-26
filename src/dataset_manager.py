import os
import random
import shutil
from config import (
    KAGGLE_GENERATED_DIR,
    ACCEPTED_DIR,
    REJECTED_DIR,
    FACIAL_PALSY_CLASSES,
    VALID_IMAGE_EXTENSIONS
)


def ensure_dataset_dirs():
    os.makedirs(KAGGLE_GENERATED_DIR, exist_ok=True)
    os.makedirs(ACCEPTED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    for class_name in FACIAL_PALSY_CLASSES:
        os.makedirs(os.path.join(ACCEPTED_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(REJECTED_DIR, class_name), exist_ok=True)


def scan_generated_dataset():

    ensure_dataset_dirs()

    dataset_summary = {}
    total_images = 0

    for class_name in FACIAL_PALSY_CLASSES:
        class_dir = os.path.join(KAGGLE_GENERATED_DIR, class_name)

        images = []

        if os.path.exists(class_dir):
            images = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
            ]

            images.sort()

        dataset_summary[class_name] = {
            "count": len(images),
            "images": images
        }

        total_images += len(images)

    return {
        "total_images": total_images,
        "classes": dataset_summary
    }


def get_class_images(class_name):
    """
    Returns images for a selected class.
    """
    class_dir = os.path.join(KAGGLE_GENERATED_DIR, class_name)

    if not os.path.exists(class_dir):
        return []

    images = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]

    images.sort()

    return [
        {
            "filename": image,
            "class_name": class_name,
            "url": f"/datasets/kaggle_generated/{class_name}/{image}"
        }
        for image in images
    ]


def move_to_accepted(class_name, filename):
    source_path = os.path.join(KAGGLE_GENERATED_DIR, class_name, filename)
    target_dir = os.path.join(ACCEPTED_DIR, class_name)
    target_path = os.path.join(target_dir, filename)

    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_path):
        return False, "Source image not found."

    shutil.copy2(source_path, target_path)
    return True, "Image copied to accepted dataset."


def move_to_rejected(class_name, filename):
    source_path = os.path.join(KAGGLE_GENERATED_DIR, class_name, filename)
    target_dir = os.path.join(REJECTED_DIR, class_name)
    target_path = os.path.join(target_dir, filename)

    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_path):
        return False, "Source image not found."

    shutil.copy2(source_path, target_path)
    return True, "Image copied to rejected dataset."


def get_preview_images(class_name="All", limit=6):

    from config import (
        KAGGLE_GENERATED_DIR,
        FACIAL_PALSY_CLASSES,
        VALID_IMAGE_EXTENSIONS
    )

    limit = int(limit)
    preview_images = []

    if class_name == "All":
        target_classes = FACIAL_PALSY_CLASSES
    else:
        target_classes = [class_name]

    print("\n=== PREVIEW IMAGE LOADING DEBUG ===")
    print("Selected class:", class_name)
    print("Images requested per class:", limit)
    print("Target classes:", target_classes)

    for cls in target_classes:
        class_dir = os.path.join(KAGGLE_GENERATED_DIR, cls)

        if not os.path.exists(class_dir):
            print(f"[MISSING FOLDER] {cls}: {class_dir}")
            continue

        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        images.sort()
        random.shuffle(images)

        selected_images = images[:limit]

        print(f"{cls}: found {len(images)} images, selected {len(selected_images)}")

        for img in selected_images:
            preview_images.append({
                "filename": img,
                "class_name": cls,
                "url": f"/datasets/kaggle_generated/{cls}/{img}"
            })

    print("Total preview images returned:", len(preview_images))
    print("===================================\n")

    random.shuffle(preview_images)

    return preview_images