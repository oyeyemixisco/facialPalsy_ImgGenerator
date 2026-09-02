import os
import csv
import shutil
from datetime import datetime

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "kaggle_generated"

from flask import Flask, render_template, request, jsonify, send_from_directory
from src.metrics import analyse_facial_palsy_image

from src.dataset_manager import (
    scan_generated_dataset,
    get_class_images,
    move_to_accepted,
    move_to_rejected
)

from src.dataset_manager import get_preview_images


app = Flask(__name__)

SESSION_ANALYSIS_RESULTS = {}

RAW_DATASET_DIR = BASE_DIR / "datasets" / "kaggle_generated"
FINAL_GRADED_DATASET_DIR = BASE_DIR / "datasets" / "final_graded_dataset"

FINAL_GRADE_CLASSES = [
    "Grade_I_Normal",
    "Grade_II_Mild",
    "Grade_III_Moderate",
    "Grade_IV_Moderate_Severe",
    "Grade_V_Severe",
    "Grade_VI_Total_Paralysis",
]

@app.route("/")
def index():
    return render_template("index.html")


def ensure_final_grade_folders():
    os.makedirs(FINAL_GRADED_DATASET_DIR, exist_ok=True)

    for grade_class in FINAL_GRADE_CLASSES:
        os.makedirs(
            os.path.join(FINAL_GRADED_DATASET_DIR, grade_class),
            exist_ok=True
        )


def get_all_raw_dataset_images():
    raw_classes = ["Healthy", "Mild_Palsy", "Moderate_Palsy", "Severe_Palsy"]

    image_records = []

    allowed_extensions = (".png", ".jpg", ".jpeg", ".webp")

    for class_name in raw_classes:
        class_dir = os.path.join(RAW_DATASET_DIR, class_name)

        if not os.path.exists(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(allowed_extensions):
                image_records.append({
                    "class_name": class_name,
                    "filename": filename,
                    "image_path": os.path.join(class_dir, filename)
                })

    return image_records


@app.route("/datasets/kaggle_generated/<class_name>/<filename>")
def serve_generated_image(class_name, filename):
    folder = DATASET_DIR / class_name
    return send_from_directory(folder, filename)


@app.route("/api/dataset/summary", methods=["GET"])
def api_dataset_summary():
    summary = scan_generated_dataset()

    return jsonify({
        "success": True,
        "summary": summary
    })


@app.route("/api/dataset/images/<class_name>", methods=["GET"])
def api_dataset_images(class_name):
    images = get_class_images(class_name)

    return jsonify({
        "success": True,
        "class_name": class_name,
        "count": len(images),
        "images": images
    })


@app.route("/api/dataset/accept", methods=["POST"])
def api_accept_image():
    data = request.get_json() or {}

    class_name = data.get("class_name")
    filename = data.get("filename")

    if not class_name or not filename:
        return jsonify({
            "success": False,
            "message": "class_name and filename are required."
        }), 400

    success, message = move_to_accepted(class_name, filename)

    return jsonify({
        "success": success,
        "message": message
    })


@app.route("/api/dataset/reject", methods=["POST"])
def api_reject_image():
    data = request.get_json() or {}

    class_name = data.get("class_name")
    filename = data.get("filename")

    if not class_name or not filename:
        return jsonify({
            "success": False,
            "message": "class_name and filename are required."
        }), 400

    success, message = move_to_rejected(class_name, filename)

    return jsonify({
        "success": success,
        "message": message
    })


@app.route("/api/dataset/preview", methods=["POST"])
def api_dataset_preview():
    data = request.get_json() or {}

    class_name = data.get("class_name", "All")
    limit = int(data.get("limit", 6))

    print("========== RENDER DATASET CHECK ==========")
    print("BASE_DIR:", BASE_DIR)
    print("DATASET_DIR:", DATASET_DIR)
    print("DATASET EXISTS:", DATASET_DIR.exists())
    
    if DATASET_DIR.exists():
        for item in DATASET_DIR.iterdir():
            print("DATASET ITEM:", item)
    
    healthy_dir = DATASET_DIR / "Healthy"
    
    print("HEALTHY DIR:", healthy_dir)
    print("HEALTHY EXISTS:", healthy_dir.exists())
    
    if healthy_dir.exists():
        files = list(healthy_dir.iterdir())
        print("HEALTHY FILE COUNT:", len(files))
        print("HEALTHY FIRST FILES:", files[:10])
    
    print("==========================================")

    images = get_preview_images(
        class_name=class_name,
        limit=limit
    )

    return jsonify({
        "success": True,
        "class_name": class_name,
        "limit": limit,
        "count": len(images),
        "images": images
    })

@app.route("/api/validate-image", methods=["POST"])
def api_validate_image():
    data = request.get_json() or {}

    class_name = data.get("class_name")
    filename = data.get("filename")

    if not class_name or not filename:
        return jsonify({
            "success": False,
            "message": "class_name and filename are required."
        }), 400

    image_path = DATASET_DIR / class_name / filename

    if not os.path.exists(image_path):
        return jsonify({
            "success": False,
            "message": f"Image not found: {image_path}"
        }), 404

    try:
        result = analyse_facial_palsy_image(
            image_path=image_path,
            class_name=class_name
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Validation failed: {str(e)}"
        }), 500
    

@app.route("/api/analyze-loaded-images", methods=["POST"])
def api_analyze_loaded_images():
    data = request.get_json() or {}

    images = data.get("images", [])

    if not images:
        return jsonify({
            "success": False,
            "message": "No images provided for analysis."
        }), 400

    accepted = 0
    review = 0
    rejected = 0
    results = []

    for img in images:
        class_name = img.get("class_name")
        filename = img.get("filename")

        if not class_name or not filename:
            continue

        image_path = DATASET_DIR / class_name / filename

        if not os.path.exists(image_path):
            continue

        try:
            result = analyse_facial_palsy_image(
                image_path=image_path,
                class_name=class_name
            )

            decision = result.get("decision", "Needs Review")
            decision_lower = decision.lower()

            if "accepted" in decision_lower:
                accepted += 1
            elif "rejected" in decision_lower:
                rejected += 1
            else:
                review += 1

            result_record = {
                "class_name": class_name,
                "filename": filename,
                "image_url": f"/datasets/kaggle_generated/{class_name}/{filename}",
                **result
            }

            SESSION_ANALYSIS_RESULTS[filename] = result_record
            results.append(result_record)

        except Exception as e:
            rejected += 1

            result_record = {
                "success": False,
                "class_name": class_name,
                "filename": filename,
                "image_url": f"/datasets/kaggle_generated/{class_name}/{filename}",
                "decision": "Rejected",
                "message": str(e)
            }

            SESSION_ANALYSIS_RESULTS[filename] = result_record
            results.append(result_record)

    return jsonify({
        "success": True,
        "summary": {
            "loaded": len(images),
            "accepted": accepted,
            "review": review,
            "rejected": rejected
        },
        "results": results
    })

@app.route("/validation-metrics")
def validation_metrics_page():
    return render_template("validation_metrics.html")


@app.route("/api/validation-detail", methods=["POST"])
def api_validation_detail():
    data = request.get_json() or {}

    class_name = data.get("class_name")
    filename = data.get("filename")

    if not class_name or not filename:
        return jsonify({
            "success": False,
            "message": "class_name and filename are required."
        }), 400

    image_path = DATASET_DIR / class_name / filename

    if not os.path.exists(image_path):
        return jsonify({
            "success": False,
            "message": f"Image not found: {image_path}"
        }), 404

    try:
        result = analyse_facial_palsy_image(
            image_path=image_path,
            class_name=class_name
        )

        result_record = {
            "class_name": class_name,
            "filename": filename,
            "image_url": f"/datasets/kaggle_generated/{class_name}/{filename}",
            **result
        }

        SESSION_ANALYSIS_RESULTS[filename] = result_record

        return jsonify({
            "success": True,
            "result": result_record
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "message": f"Validation failed: {str(e)}"
        }), 500
    

@app.route("/api/export-final-graded-dataset", methods=["POST"])
def export_final_graded_dataset():
    ensure_final_grade_folders()

    images = get_all_raw_dataset_images()

    if not images:
        return jsonify({
            "success": False,
            "message": "No images found in the raw 4-class dataset."
        }), 404

    csv_path = os.path.join(FINAL_GRADED_DATASET_DIR, "final_graded_dataset.csv")

    grade_counts = {
        "Grade_I_Normal": 0,
        "Grade_II_Mild": 0,
        "Grade_III_Moderate": 0,
        "Grade_IV_Moderate_Severe": 0,
        "Grade_V_Severe": 0,
        "Grade_VI_Total_Paralysis": 0,
    }

    exported_records = []
    failed_records = []

    for item in images:
        original_class = item["class_name"]
        filename = item["filename"]
        image_path = item["image_path"]

        try:
            result = analyse_facial_palsy_image(
                image_path=image_path,
                class_name=original_class
            )

            predicted_grade = result.get("predicted_grade", "Unknown")

            if predicted_grade not in FINAL_GRADE_CLASSES:
                failed_records.append({
                    "filename": filename,
                    "original_class": original_class,
                    "reason": f"Invalid predicted grade: {predicted_grade}"
                })
                continue

            destination_folder = os.path.join(
                FINAL_GRADED_DATASET_DIR,
                predicted_grade
            )

            os.makedirs(destination_folder, exist_ok=True)

            new_filename = f"{original_class}_{filename}"

            destination_path = os.path.join(destination_folder, new_filename)

            shutil.copy2(image_path, destination_path)

            grade_counts[predicted_grade] += 1

            record = {
                "filename": filename,
                "exported_filename": new_filename,
                "original_class": original_class,
                "predicted_severity": result.get("predicted_severity", ""),
                "predicted_grade": predicted_grade,
                "predicted_grade_display": result.get("predicted_grade_display", ""),
                "composite_score": result.get("composite_score", ""),
                "mouth_asymmetry": result.get("mouth_asymmetry", ""),
                "eye_asymmetry": result.get("eye_asymmetry", ""),
                "eyebrow_asymmetry": result.get("eyebrow_asymmetry", ""),
                "global_symmetry_error": result.get("global_symmetry_error", ""),
                "detected_side": result.get("detected_side", ""),
                "side_confidence": result.get("side_confidence", ""),
                "status": result.get("decision", ""),
                "source_path": image_path,
                "final_path": destination_path,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            exported_records.append(record)

        except Exception as e:
            failed_records.append({
                "filename": filename,
                "original_class": original_class,
                "reason": str(e)
            })

    fieldnames = [
        "filename",
        "exported_filename",
        "original_class",
        "predicted_severity",
        "predicted_grade",
        "predicted_grade_display",
        "composite_score",
        "mouth_asymmetry",
        "eye_asymmetry",
        "eyebrow_asymmetry",
        "global_symmetry_error",
        "detected_side",
        "side_confidence",
        "status",
        "source_path",
        "final_path",
        "exported_at"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in exported_records:
            writer.writerow(record)

    return jsonify({
        "success": True,
        "message": "Final 6-grade dataset exported successfully.",
        "total_images_found": len(images),
        "total_exported": len(exported_records),
        "total_failed": len(failed_records),
        "grade_counts": grade_counts,
        "csv_path": csv_path,
        "failed_records": failed_records[:20]
    })

@app.route("/datasets/final_graded_dataset/<grade_name>/<filename>")
def serve_final_graded_dataset_image(grade_name, filename):
    return send_from_directory(
        os.path.join(FINAL_GRADED_DATASET_DIR, grade_name),
        filename
    )

if __name__ == "__main__":
    app.run(debug=True)
