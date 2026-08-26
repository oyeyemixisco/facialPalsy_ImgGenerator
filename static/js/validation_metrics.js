function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}


function applyDetailStatusColour(statusText) {
  const statusEl = document.getElementById("detailStatus");

  if (!statusEl) return;

  statusEl.classList.remove(
    "status-accepted",
    "status-rejected",
    "status-review",
    "status-neutral"
  );

  const status = (statusText || "").toLowerCase();

  if (status.includes("accepted")) {
    statusEl.classList.add("status-accepted");
  } else if (status.includes("rejected")) {
    statusEl.classList.add("status-rejected");
  } else if (
    status.includes("needs review") ||
    status.includes("review") ||
    status.includes("borderline") ||
    status.includes("mismatch")
  ) {
    statusEl.classList.add("status-review");
  } else {
    statusEl.classList.add("status-neutral");
  }
}


async function loadValidationDetails() {
  const params = new URLSearchParams(window.location.search);

  const className = params.get("class_name");
  const filename = params.get("filename");

  if (!className || !filename) {
    alert("Missing class name or filename. Please go back and select an image again.");
    return;
  }

  setText("detailClass", className);
  setText("detailStatus", "Analysing...");
  setText("detailSeverity", "Analysing...");
  setText("detailPredictedGrade", "Checking...");
  setText("detailFaceDetected", "Checking...");
  setText("detailExpectedSide", "Checking...");
  setText("detailDetectedSide", "Checking...");
  setText("detailSideConfidence", "Checking...");
  setText("detailLandmarks", "Checking...");
  setText("detailResolution", "Checking...");
  setText("detailMouth", "...");
  setText("detailEye", "...");
  setText("detailBrow", "...");
  setText("detailGlobal", "...");
  setText("detailComposite", "...");
  setText("detailRange", "...");

  applyDetailStatusColour("neutral");

  try {
    const response = await fetch("/api/validation-detail", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        class_name: className,
        filename: filename
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
        const message = data.message || "Could not load validation details.";

        alert(message);

        setText("detailStatus", "Validation Error");
        setText("detailSeverity", "—");
        setText("detailFaceDetected", data.face_detected || "—");
        setText("detailLandmarks", data.landmarks || "—");
        setText("detailResolution", data.resolution || "—");
        setText("detailMouth", "—");
        setText("detailEye", "—");
        setText("detailBrow", "—");
        setText("detailGlobal", "—");
        setText("detailComposite", "—");
        setText("detailRange", "—");

        applyDetailStatusColour("review");

        return;
    }

    const result = data.result;

    const image = document.getElementById("detailOverlayImage");

    if (image) {
      image.src = result.overlay_image_url || result.image_url;
    }

    setText("detailClass", result.class_name || className);
    setText("detailStatus", result.decision || "—");
    setText("detailSeverity", result.predicted_severity || "—");
    setText("detailPredictedGrade", result.predicted_grade_display || result.predicted_grade || "—");
    setText("detailDetectedSide", result.detected_side || "—");
    setText("detailQualityNote", result.quality_note || result.expression_reason || "—");
    setText("detailSideConfidence", formatConfidence(result.side_confidence));
    setText("detailFaceDetected", result.face_detected || "—");
    setText("detailLandmarks", result.landmarks || "—");
    setText("detailResolution", result.resolution || "—");

    setText("detailMouth", result.mouth_asymmetry ?? "—");
    setText("detailEye", result.eye_asymmetry ?? "—");
    setText("detailBrow", result.eyebrow_asymmetry ?? "—");
    setText("detailGlobal", result.global_symmetry_error ?? "—");
    setText("detailComposite", result.composite_palsy_score ?? "—");
    setText("detailRange", result.expected_range || "—");

    applyDetailStatusColour(result.decision || "neutral");

  } catch (error) {
    console.error("Validation details error:", error);
    alert("Validation details could not be loaded. Check Flask console for the backend error.");
    setText("detailStatus", "Validation error");
    applyDetailStatusColour("rejected");
  }
}


function formatConfidence(value) {
  if (value === null || value === undefined || value === "—") return "—";

  const number = Number(value);
  if (Number.isNaN(number)) return value;

  return `${(number * 100).toFixed(2)}%`;
}


document.addEventListener("DOMContentLoaded", loadValidationDetails);