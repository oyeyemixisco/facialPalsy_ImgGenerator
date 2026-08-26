let distributionChart = null;
let selectedImageData = null;

const TOTAL_CLASSES = 4;

let loadedImagesForSession = [];
let analysisResultsByFilename = {};

const classDisplayNames = {
  All: "all facial palsy classes",
  Healthy: "healthy patients with normal facial symmetry",
  Mild_Palsy: "mild facial palsy",
  Moderate_Palsy: "moderate facial palsy",
  Severe_Palsy: "severe facial palsy"
};

const metricProfiles = {
  Healthy: {
    status: "Valid",
    faceDetected: "Yes",
    landmarks: "468/468",
    resolution: "512 × 512",
    mouth: "0.032",
    eye: "0.028",
    brow: "0.026",
    global: "0.031",
    composite: "0.030",
    range: "0.00 - 0.08"
  },

  Mild_Palsy: {
    status: "Valid",
    faceDetected: "Yes",
    landmarks: "468/468",
    resolution: "512 × 512",
    mouth: "0.116",
    eye: "0.104",
    brow: "0.091",
    global: "0.108",
    composite: "0.106",
    range: "0.08 - 0.20"
  },

  Moderate_Palsy: {
    status: "Valid",
    faceDetected: "Yes",
    landmarks: "468/468",
    resolution: "512 × 512",
    mouth: "0.286",
    eye: "0.241",
    brow: "0.216",
    global: "0.253",
    composite: "0.249",
    range: "0.20 - 0.40"
  },

  Severe_Palsy: {
    status: "Valid",
    faceDetected: "Yes",
    landmarks: "468/468",
    resolution: "512 × 512",
    mouth: "0.512",
    eye: "0.467",
    brow: "0.438",
    global: "0.481",
    composite: "0.475",
    range: "0.40 - 1.00"
  }

};


let validationCounters = {
  loaded: 0,
  accepted: 0,
  review: 0,
  rejected: 0,
  validatedFiles: {}
};


/* =========================
   Utility Functions
========================= */

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}


function setBar(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.style.width = `${value}%`;
  }
}


function safeNumber(value, fallback = 0) {
  const parsed = parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}


/* =========================
   Theme Toggle
========================= */

function applyTheme(theme) {
  const body = document.body;
  const themeIcon = document.getElementById("themeIcon");
  const themeText = document.getElementById("themeText");

  if (theme === "light") {
    body.classList.add("light-theme");

    if (themeIcon) themeIcon.textContent = "🌙";
    if (themeText) themeText.textContent = "Dark";
  } else {
    body.classList.remove("light-theme");

    if (themeIcon) themeIcon.textContent = "☀️";
    if (themeText) themeText.textContent = "Light";
  }

  localStorage.setItem("dashboardTheme", theme);
}


function toggleTheme() {
  const isLight = document.body.classList.contains("light-theme");
  applyTheme(isLight ? "dark" : "light");
}


function initialiseTheme() {
  const savedTheme = localStorage.getItem("dashboardTheme");

  if (savedTheme) {
    applyTheme(savedTheme);
  } else {
    applyTheme("light");
  }

  const themeToggleBtn = document.getElementById("themeToggleBtn");

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", toggleTheme);
  }
}


/* =========================
   Prompt / Control Panel
========================= */

function updatePromptText() {
  const classSelect = document.getElementById("classSelect");
  const imageCount = document.getElementById("imageCount");
  const promptInput = document.getElementById("promptInput");

  const targetImagesText = document.getElementById("targetImagesText");
  const selectedClassText = document.getElementById("selectedClassText");
  const totalToLoadText = document.getElementById("totalToLoadText");

  if (!classSelect || !imageCount || !promptInput) return;

  const selectedClass = classSelect.value;
  const count = safeNumber(imageCount.value, 1);

  if (selectedClass === "All") {
    const totalImages = count * TOTAL_CLASSES;

    promptInput.value =
      `Generate ${count} frontal clinical facial images for each facial palsy class`;

    setText("targetImagesText", `${count} per class`);
    setText("selectedClassText", "All Classes");
    setText("totalToLoadText", totalImages);
  } else if (selectedClass === "Healthy") {
    promptInput.value =
      `Generate ${count} frontal clinical facial images of healthy patients with normal facial symmetry.`;

    setText("targetImagesText", count);
    setText("selectedClassText", "Healthy");
    setText("totalToLoadText", count);
  } else {
    const readableClass = classDisplayNames[selectedClass] || selectedClass;

    promptInput.value =
      `Generate ${count} frontal clinical facial images of patients with ${readableClass}.`;

    setText("targetImagesText", count);
    setText("selectedClassText", readableClass);
    setText("totalToLoadText", count);
  }
}


/* =========================
   Loader
========================= */

function showLoader(
  title = "Generating in progress...",
  message = "Preparing synthetic image preview"
) {
  const overlay = document.getElementById("loaderOverlay");
  const progress = document.getElementById("loaderProgress");
  const loaderTitle = document.getElementById("loaderTitle");
  const loaderText = document.getElementById("loaderText");

  if (!overlay || !progress) return null;

  overlay.classList.remove("hidden");
  progress.style.width = "0%";

  if (loaderTitle) {
    loaderTitle.textContent = title;
  }

  if (loaderText) {
    loaderText.textContent = message;
  }

  let value = 0;

  const interval = setInterval(() => {
    value += Math.floor(Math.random() * 12) + 5;

    if (value >= 95) {
      value = 95;
      clearInterval(interval);
    }

    progress.style.width = `${value}%`;
  }, 250);

  return interval;
}

function hideLoader(interval) {
  const overlay = document.getElementById("loaderOverlay");
  const progress = document.getElementById("loaderProgress");

  if (interval) {
    clearInterval(interval);
  }

  if (progress) {
    progress.style.width = "100%";
  }

  setTimeout(() => {
    if (overlay) {
      overlay.classList.add("hidden");
    }

    if (progress) {
      progress.style.width = "0%";
    }
  }, 600);
}


/* =========================
   Generated Dataset Preview
========================= */

async function loadGeneratedPreviewImages() {
  const classSelect = document.getElementById("classSelect");
  const imageCount = document.getElementById("imageCount");
  const previewGrid = document.getElementById("generatedPreviewGrid");
  const loadedImageCount = document.getElementById("loadedImageCount");
  const loaderText = document.getElementById("loaderText");

  if (!classSelect || !imageCount || !previewGrid) return;

  const selectedClass = classSelect.value;
  const count = safeNumber(imageCount.value, 5);

  updatePromptText();
  resetSelectedImagePanel();

  if (loadedImageCount) {
    loadedImageCount.textContent = "Loading images...";
  }

  if (loaderText) {
    if (selectedClass === "All") {
      loaderText.textContent = `Loading ${count * TOTAL_CLASSES} images across all classes...`;
    } else {
      loaderText.textContent = `Loading ${count} images from ${classDisplayNames[selectedClass] || selectedClass}...`;
    }
  }

  const loaderInterval = showLoader(
    "Generating in progress...",
    "Preparing synthetic image preview..."
  );

  previewGrid.classList.remove("empty-preview");
  previewGrid.innerHTML = "";

  try {
    const response = await fetch("/api/dataset/preview", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        class_name: selectedClass,
        limit: count
      })
    });

    const data = await response.json();

    await new Promise(resolve => setTimeout(resolve, 1000));

    hideLoader(loaderInterval);

    if (!data.success || !data.images || data.images.length === 0) {

      previewGrid.classList.add("empty-preview");
      previewGrid.innerHTML = `<p>No images found for the selected class.</p>`;

      loadedImagesForSession = [];

      if (loadedImageCount) {
        loadedImageCount.textContent = "0 images loaded";
      }

      updateStatsAfterLoad(0);
      return;
    }

    loadedImagesForSession = data.images;
    analysisResultsByFilename = {};

    previewGrid.classList.remove("empty-preview");
    previewGrid.innerHTML = "";

    data.images.forEach((img, index) => {
      const card = document.createElement("div");
      card.className = "generated-preview-card";
      card.dataset.filename = img.filename;

      card.innerHTML = `
        <img src="${img.url}" alt="${img.filename}">
        <div class="generated-preview-info">
          <span>${img.class_name}</span>
        </div>
      `;

      card.addEventListener("click", () => {
        selectGeneratedImage(img, card);
      });

      previewGrid.appendChild(card);

      if (index === 0) {
        setTimeout(() => {
          selectGeneratedImage(img, card);
        }, 100);
      }
    });

    if (loadedImageCount) {
      loadedImageCount.textContent = `${data.images.length} images loaded`;
    }

    updateStatsAfterLoad(data.images.length);
    updateProgressAfterLoad(data.images.length);

  } catch (error) {
    console.error("Error loading preview images:", error);

    hideLoader(loaderInterval);

    previewGrid.classList.add("empty-preview");
    previewGrid.innerHTML = `<p>Error loading generated images.</p>`;

    if (loadedImageCount) {
      loadedImageCount.textContent = "0 images loaded";
    }

    updateStatsAfterLoad(0);
  }
}


function selectGeneratedImage(img, selectedCard) {
  selectedImageData = img;

  document.querySelectorAll(".generated-preview-card").forEach(card => {
    card.classList.remove("selected");
  });

  if (selectedCard) {
    selectedCard.classList.add("selected");
  }

  const selectedImageBox = document.getElementById("selectedImageBox");
  const selectedImage = document.getElementById("selectedImage");

  if (selectedImageBox) {
    selectedImageBox.classList.remove("empty-selected-box");

    const placeholder = selectedImageBox.querySelector("p");
    if (placeholder) {
      placeholder.style.display = "none";
    }
  }

  if (selectedImage) {
    selectedImage.src = img.url;   // plain image first
    selectedImage.classList.remove("hidden");
  }

  document.querySelectorAll(".midline, .hline").forEach(line => {
    line.classList.add("hidden");
  });

  setText("selectedClassName", classDisplayNames[img.class_name] || img.class_name);
  setText("imageStatus", "Not analysed");
  setText("predictedSeverity", "—");
  setText("faceDetected", "—");
  setText("landmarksCount", "—");
  setText("selectedResolution", "—");
  setText("mouthMetric", "—");
  setText("eyeMetric", "—");
  setText("browMetric", "—");
  setText("globalMetric", "—");
  setText("compositeMetric", "—");
  setText("expectedRange", "—");

  applyValidationStatusColour("neutral");

  const existingResult = analysisResultsByFilename[img.filename];

  if (existingResult) {
    populateMetricsFromResult(existingResult);
  } else {
    setText("imageStatus", "Not analysed");
    setText("predictedSeverity", "—");
    setText("faceDetected", "—");
    setText("landmarksCount", "—");
    setText("selectedResolution", "—");
    setText("mouthMetric", "—");
    setText("eyeMetric", "—");
    setText("browMetric", "—");
    setText("globalMetric", "—");
    setText("compositeMetric", "—");
    setText("expectedRange", "—");
    applyValidationStatusColour("neutral");
  }

}


function resetSelectedImagePanel() {
  selectedImageData = null;

  const selectedImageBox = document.getElementById("selectedImageBox");
  const selectedImage = document.getElementById("selectedImage");

  if (selectedImageBox) {
    selectedImageBox.classList.add("empty-selected-box");

    const placeholder = selectedImageBox.querySelector("p");
    if (placeholder) {
      placeholder.style.display = "block";
      placeholder.textContent = "No image selected";
    }
  }

  if (selectedImage) {
    selectedImage.src = "";
    selectedImage.classList.add("hidden");
  }

  document.querySelectorAll(".midline, .hline").forEach(line => {
    line.classList.add("hidden");
  });

  setText("selectedClassName", "—");
  setText("imageStatus", "Not analysed");
  applyValidationStatusColour("neutral");
  setText("predictedSeverity", "—");
  setText("predictedGradeText", "—");
  setText("faceDetected", "—");
  setText("landmarksCount", "—");
  setText("selectedResolution", "—");
  setText("mouthMetric", "—");
  setText("eyeMetric", "—");
  setText("browMetric", "—");
  setText("globalMetric", "—");
  setText("compositeMetric", "—");
  setText("expectedRange", "—");

  const imageStatus = document.getElementById("imageStatus");
  if (imageStatus) {
    imageStatus.classList.remove("valid-text");
  }

  document.querySelectorAll(".generated-preview-card").forEach(card => {
    card.classList.remove("selected");
  });
}


function clearPreview() {
  const previewGrid = document.getElementById("generatedPreviewGrid");
  const loadedImageCount = document.getElementById("loadedImageCount");

  if (previewGrid) {
    previewGrid.classList.add("empty-preview");
    previewGrid.innerHTML = `<p>No generated images loaded yet.</p>`;
  }

  if (loadedImageCount) {
    loadedImageCount.textContent = "0 images loaded";
  }

  validationCounters = {
    loaded: 0,
    accepted: 0,
    review: 0,
    rejected: 0,
    validatedFiles: {}
  };

  resetSelectedImagePanel();
  refreshDashboardStats();
  updateProgressAfterLoad(0);
}


/* =========================
   Temporary Metrics
   Replace later with backend validation
========================= */

function populateTemporaryMetrics(className) {
  const profile = metricProfiles[className] || metricProfiles.Healthy;

  setText("imageStatus", profile.status);
  setText("faceDetected", profile.faceDetected);
  setText("landmarksCount", profile.landmarks);
  setText("selectedResolution", profile.resolution);
  setText("mouthMetric", profile.mouth);
  setText("eyeMetric", profile.eye);
  setText("browMetric", profile.brow);
  setText("globalMetric", profile.global);
  setText("compositeMetric", profile.composite);
  setText("expectedRange", profile.range);

}

function applyValidationStatusColour(statusText) {
  const imageStatus = document.getElementById("imageStatus");

  if (!imageStatus) return;

  imageStatus.classList.remove(
    "status-accepted",
    "status-rejected",
    "status-review",
    "status-neutral",
    "valid-text",
    "error-text"
  );

  const status = (statusText || "").toLowerCase();

  if (status.includes("accepted")) {
    imageStatus.classList.add("status-accepted");
  } else if (status.includes("rejected")) {
    imageStatus.classList.add("status-rejected");
  } else if (
    status.includes("needs review") ||
    status.includes("review") ||
    status.includes("borderline") ||
    status.includes("mismatch")
  ) {
    imageStatus.classList.add("status-review");
  } else {
    imageStatus.classList.add("status-neutral");
  }
}

async function runValidationForSelectedImage() {
  if (!selectedImageData) {
    alert("Please select an image first.");
    return;
  }

  setText("imageStatus", "Analysing...");
  applyValidationStatusColour("neutral");
  setText("predictedSeverity", "Analysing...");
  setText("faceDetected", "Checking...");
  setText("landmarksCount", "Checking...");
  setText("selectedResolution", "Checking...");
  setText("mouthMetric", "...");
  setText("eyeMetric", "...");
  setText("browMetric", "...");
  setText("globalMetric", "...");
  setText("compositeMetric", "...");
  setText("expectedRange", "...");

  try {
    const response = await fetch("/api/validate-image", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        class_name: selectedImageData.class_name,
        filename: selectedImageData.filename
      })
    });

    const data = await response.json();

    if (!data.success) {
      setText("imageStatus", "Validation failed");
      applyValidationStatusColour("Rejected");
      setText("faceDetected", data.face_detected || "No");
      setText("landmarksCount", data.landmarks || "0/468");
      setText("selectedResolution", data.resolution || "—");

      alert(data.message || "Validation failed.");
      return;
    }

    setText("imageStatus", data.decision);
    setText("predictedSeverity", data.predicted_severity || "—");
    applyValidationStatusColour(data.decision);
    updateValidationCounters(selectedImageData.filename, data.decision);
    updateSelectedCardStatus(data.decision);
    setText("faceDetected", data.face_detected);
    setText("landmarksCount", data.landmarks);
    setText("selectedResolution", data.resolution);
    setText("mouthMetric", data.mouth_asymmetry);
    setText("eyeMetric", data.eye_asymmetry);
    setText("browMetric", data.eyebrow_asymmetry);
    setText("globalMetric", data.global_symmetry_error);
    setText("compositeMetric", data.composite_palsy_score);
    setText("expectedRange", data.expected_range);

    const selectedImage = document.getElementById("selectedImage");

    if (selectedImage && data.overlay_image_url) {
      selectedImage.src = data.overlay_image_url;
    }

    const imageStatus = document.getElementById("imageStatus");

    if (imageStatus) {
      imageStatus.classList.remove("valid-text", "error-text");

      if (data.decision === "Accepted") {
        imageStatus.classList.add("valid-text");
      } else {
        imageStatus.classList.add("error-text");
      }
    }

  } catch (error) {
    console.error("Validation error:", error);
    setText("imageStatus", "Validation error");
    alert("Could not validate this image.");
  }
}


function updateSelectedCardStatus(decision) {
  const selectedCard = document.querySelector(".generated-preview-card.selected");
  if (!selectedCard) return;

  const statusSpan = selectedCard.querySelector(".generated-preview-info span");
  if (!statusSpan) return;

  statusSpan.classList.remove("status-accepted", "status-rejected", "status-review");

  const decisionLower = decision.toLowerCase();

  if (decisionLower.includes("accepted")) {
    statusSpan.textContent = "Valid";
    statusSpan.classList.add("status-accepted");
  } else if (decisionLower.includes("rejected")) {
    statusSpan.textContent = "Rejected";
    statusSpan.classList.add("status-rejected");
  } else {
    statusSpan.textContent = "Review";
    statusSpan.classList.add("status-review");
  }
}


/* Stats / Progress / Footer */

function createStats(state) {
  const grid = document.getElementById("statsGrid");
  if (!grid) return;

  const statConfig = [
    { key: "total_classes", label: "Total Classes", meta: "Facial Conditions" },
    { key: "images_per_class", label: "Images per Class", meta: "Target" },
    { key: "total_images", label: "Total Images", meta: "Loaded Dataset Size" },
    { key: "generated_images", label: "Loaded Images", metaKey: "completion_percent", metaSuffix: "% Loaded" },
    { key: "valid_images", label: "Valid Images", metaKey: "valid_percent", metaSuffix: "% Valid" },
    { key: "review_images", label: "Needs Review", metaKey: "review_percent", metaSuffix: "% Review" },
    { key: "rejected_images", label: "Rejected Images", metaKey: "rejected_percent", metaSuffix: "% Rejected" },
  ];

  grid.innerHTML = "";

  statConfig.forEach(item => {
    const metaText = item.metaKey
      ? `${state[item.metaKey]}${item.metaSuffix}`
      : item.meta;

    const card = document.createElement("div");
    card.className = "stat-card";
    card.innerHTML = `
      <div class="stat-label">${item.label}</div>
      <div class="stat-value">${state[item.key]}</div>
      <div class="stat-meta">${metaText}</div>
    `;

    grid.appendChild(card);
  });
}


function updateStatsAfterLoad(loadedCount) {
  validationCounters.loaded = safeNumber(loadedCount, 0);
  validationCounters.accepted = 0;
  validationCounters.review = 0;
  validationCounters.rejected = 0;
  validationCounters.validatedFiles = {};

  refreshDashboardStats();
}

function refreshDashboardStats() {
  const loaded = validationCounters.loaded;
  const accepted = validationCounters.accepted;
  const review = validationCounters.review;
  const rejected = validationCounters.rejected;

  const validPercent = loaded > 0 ? ((accepted / loaded) * 100).toFixed(2) : "0";
  const reviewPercent = loaded > 0 ? ((review / loaded) * 100).toFixed(2) : "0";
  const rejectedPercent = loaded > 0 ? ((rejected / loaded) * 100).toFixed(2) : "0";

  const state = {
    total_classes: TOTAL_CLASSES,
    images_per_class: document.getElementById("imageCount")
      ? document.getElementById("imageCount").value
      : 0,

    total_images: loaded,
    generated_images: loaded,
    valid_images: accepted,
    review_images: review,
    rejected_images: rejected,

    completion_percent: loaded > 0 ? "100" : "0",
    valid_percent: validPercent,
    review_percent: reviewPercent,
    rejected_percent: rejectedPercent
  };

  createStats(state);
}


function updateValidationCounters(filename, decision) {
  if (!filename || !decision) return;

  const previousDecision = validationCounters.validatedFiles[filename];

  // Remove old count if this image was already validated before
  if (previousDecision) {
    if (previousDecision === "accepted") validationCounters.accepted -= 1;
    if (previousDecision === "review") validationCounters.review -= 1;
    if (previousDecision === "rejected") validationCounters.rejected -= 1;
  }

  const decisionLower = decision.toLowerCase();

  let decisionType = "review";

  if (decisionLower.includes("accepted")) {
    decisionType = "accepted";
    validationCounters.accepted += 1;
  } else if (decisionLower.includes("rejected")) {
    decisionType = "rejected";
    validationCounters.rejected += 1;
  } else {
    decisionType = "review";
    validationCounters.review += 1;
  }

  validationCounters.validatedFiles[filename] = decisionType;

  refreshDashboardStats();
}


function updateProgressAfterLoad(loadedCount) {
  const count = safeNumber(loadedCount, 0);
  const percent = count > 0 ? 100 : 0;

  setText("overallProgressText", `${percent.toFixed(2)}%`);
  setText("classProgressText", `${percent.toFixed(2)}%`);
  setText("elapsedTime", count > 0 ? "00:00:04" : "00:00:00");
  setText("remainingTime", "00:00:00");
  setText("imagesPerMin", count > 0 ? count : "0");

  setBar("overallProgressBar", percent);
  setBar("classProgressBar", percent);
}


function updateSideStatus() {
  setText("gpuName", "Kaggle SDXL-LoRA");
  setText("gpuUsageValue", "Offline");
  setText("ramUsageValue", "Local");
  setText("diskUsageValue", "Ready");

  setBar("gpuBar", 37);
  setBar("ramBar", 39);
  setBar("diskBar", 25);
}


function updateFooter() {
  setText("modelName", "Model: SDXL img2img + Facial Palsy LoRA");
  setText("resolutionValue", "Resolution: 512x512");
  setText("guidanceScale", "Guidance Scale: 8.5");
  setText("stepsValue", "Steps: Imported Dataset");
  setText("connectionState", "Connected");
}


/* =========================
   Distribution Chart
========================= */

function renderDistributionChart(distribution) {
  const canvas = document.getElementById("distributionChart");
  if (!canvas || typeof Chart === "undefined") return;

  const ctx = canvas.getContext("2d");
  const labels = distribution.map(item => item.label);
  const data = distribution.map(item => item.value);

  if (distributionChart) {
    distributionChart.destroy();
  }

  const isLight = document.body.classList.contains("light-theme");
  const legendColor = isLight ? "#0f172a" : "#d9e7ff";

  distributionChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: [
          "#93d36b",
          "#68ace6",
          "#f2a03b",
          "#5bc7c1",
          "#d97eb8",
          "#7c6ed8",
          "#f87171"
        ],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: {
            color: legendColor,
            boxWidth: 14
          }
        }
      }
    }
  });
}


function renderDefaultChart() {
  const distribution = [
    { label: "Healthy", value: 0 },
    { label: "Mild Palsy", value: 0 },
    { label: "Moderate Palsy", value: 0 },
    { label: "Severe Palsy", value: 0 }
  ];

  renderDistributionChart(distribution);
}


function updateDistributionFromLoadedImages() {
  const cards = document.querySelectorAll(".generated-preview-card");
  const counts = {
    Healthy: 0,
    Mild_Palsy: 0,
    Moderate_Palsy: 0,
    Severe_Palsy: 0
  };

  cards.forEach(card => {
    const classLabel = card.querySelector(".generated-preview-info span");
    if (classLabel && counts[classLabel.textContent] !== undefined) {
      counts[classLabel.textContent] += 1;
    }
  });

  const distribution = [
    { label: "Healthy", value: counts.Healthy },
    { label: "Mild Palsy", value: counts.Mild_Palsy },
    { label: "Moderate Palsy", value: counts.Moderate_Palsy },
    { label: "Severe Palsy", value: counts.Severe_Palsy }
  ];

  renderDistributionChart(distribution);
}


async function analyzeLoadedImages() {
  if (!loadedImagesForSession || loadedImagesForSession.length === 0) {
    alert("Please load images first.");
    return;
  }

  const loadedImageCount = document.getElementById("loadedImageCount");

  if (loadedImageCount) {
    loadedImageCount.textContent = "Analysing images...";
  }

  const loaderInterval = showLoader(
    "Validation in progress...",
    `Analysing ${loadedImagesForSession.length} loaded images...`
  );
  const loaderText = document.getElementById("loaderText");

  if (loaderText) {
    loaderText.textContent = `Analysing ${loadedImagesForSession.length} loaded images...`;
  }

  try {
    const response = await fetch("/api/analyze-loaded-images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        images: loadedImagesForSession
      })
    });

    const data = await response.json();

    hideLoader(loaderInterval);

    if (!data.success) {
      alert(data.message || "Analysis failed.");
      return;
    }

    analysisResultsByFilename = {};

    data.results.forEach(result => {
      analysisResultsByFilename[result.filename] = result;
    });

    updateDashboardCountersFromSummary(data.summary);
    updatePreviewCardsAfterBatchAnalysis(data.results);

    if (loadedImageCount) {
      loadedImageCount.textContent = `${data.summary.loaded} images analysed`;
    }

  } catch (error) {
    console.error(error);
    hideLoader(loaderInterval);
    alert("Could not analyse loaded images.");
  }
}

function updateDashboardCountersFromSummary(summary) {
  const loaded = summary.loaded || 0;
  const accepted = summary.accepted || 0;
  const review = summary.review || 0;
  const rejected = summary.rejected || 0;

  const validPercent = loaded > 0 ? ((accepted / loaded) * 100).toFixed(2) : "0";
  const reviewPercent = loaded > 0 ? ((review / loaded) * 100).toFixed(2) : "0";
  const rejectedPercent = loaded > 0 ? ((rejected / loaded) * 100).toFixed(2) : "0";

  const state = {
    total_classes: TOTAL_CLASSES,
    images_per_class: document.getElementById("imageCount")
      ? document.getElementById("imageCount").value
      : 0,

    total_images: loaded,
    generated_images: loaded,
    valid_images: accepted,
    review_images: review,
    rejected_images: rejected,

    completion_percent: loaded > 0 ? "100" : "0",
    valid_percent: validPercent,
    review_percent: reviewPercent,
    rejected_percent: rejectedPercent
  };

  createStats(state);
}

function updatePreviewCardsAfterBatchAnalysis(results) {
  results.forEach(result => {
    const card = document.querySelector(
      `.generated-preview-card[data-filename="${result.filename}"]`
    );

    if (!card) return;

    const statusSpan = card.querySelector(".generated-preview-info span");
    if (!statusSpan) return;

    const decision = (result.decision || "").toLowerCase();

    statusSpan.classList.remove(
      "status-accepted",
      "status-review",
      "status-rejected"
    );

    if (decision.includes("accepted")) {
      statusSpan.textContent = "Valid";
      statusSpan.classList.add("status-accepted");
    } else if (decision.includes("rejected")) {
      statusSpan.textContent = "Rejected";
      statusSpan.classList.add("status-rejected");
    } else {
      statusSpan.textContent = "Review";
      statusSpan.classList.add("status-review");
    }
  });
}

function populateMetricsFromResult(data) {
  setText("imageStatus", data.decision || "—");
  setText("predictedSeverity", data.predicted_severity || "—");
  applyValidationStatusColour(data.decision || "neutral");
  setText("predictedGradeText", data.predicted_grade_display || data.predicted_grade || "—");

  setText("faceDetected", data.face_detected || "—");
  setText("landmarksCount", data.landmarks || "—");
  setText("selectedResolution", data.resolution || "—");
  setText("mouthMetric", data.mouth_asymmetry ?? "—");
  setText("eyeMetric", data.eye_asymmetry ?? "—");
  setText("browMetric", data.eyebrow_asymmetry ?? "—");
  setText("globalMetric", data.global_symmetry_error ?? "—");
  setText("compositeMetric", data.composite_palsy_score ?? "—");
  setText("expectedRange", data.expected_range || "—");
}

function openValidationDetailsPage() {
  if (!selectedImageData) {
    alert("Please select an image first.");
    return;
  }

  const filename = encodeURIComponent(selectedImageData.filename);
  const className = encodeURIComponent(selectedImageData.class_name);

  const url = `/validation-metrics?class_name=${className}&filename=${filename}`;

  window.location.href = url;
}


async function exportFinalGradedDataset() {
  const exportBtn = document.getElementById("exportFinalDatasetBtn");

  if (exportBtn) {
    exportBtn.disabled = true;
    exportBtn.textContent = "Exporting...";
  }

  try {
    const response = await fetch("/api/export-final-graded-dataset", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      alert(data.message || "Failed to export final graded dataset.");
      return;
    }

    const counts = data.grade_counts || {};

    const summaryMessage =
      `Final 6-grade dataset exported successfully.\n\n` +
      `Total images found: ${data.total_images_found}\n` +
      `Total exported: ${data.total_exported}\n` +
      `Total failed: ${data.total_failed}\n\n` +
      `Grade I: ${counts.Grade_I_Normal || 0}\n` +
      `Grade II: ${counts.Grade_II_Mild || 0}\n` +
      `Grade III: ${counts.Grade_III_Moderate || 0}\n` +
      `Grade IV: ${counts.Grade_IV_Moderate_Severe || 0}\n` +
      `Grade V: ${counts.Grade_V_Severe || 0}\n` +
      `Grade VI: ${counts.Grade_VI_Total_Paralysis || 0}\n\n` +
      `CSV saved at:\n${data.csv_path}`;

    alert(summaryMessage);

  } catch (error) {
    console.error(error);
    alert("Unexpected error while exporting final graded dataset.");
  } finally {
    if (exportBtn) {
      exportBtn.disabled = false;
      exportBtn.textContent = "Export Final 6-Grade Dataset";
    }
  }
}


/* =========================
   Initialise Dashboard
========================= */

document.addEventListener("DOMContentLoaded", () => {
  initialiseTheme();

  const classSelect = document.getElementById("classSelect");
  const imageCount = document.getElementById("imageCount");
  const generateBtn = document.getElementById("generatePreviewBtn");
  const clearBtn = document.getElementById("clearPreviewBtn");
  const analyseBtn = document.getElementById("analyseImageBtn");

  updatePromptText();
  clearPreview();
  updateSideStatus();
  updateFooter();
  renderDefaultChart();

  const analyzeLoadedImagesBtn = document.getElementById("analyzeLoadedImagesBtn");

  if (analyzeLoadedImagesBtn) {
    analyzeLoadedImagesBtn.addEventListener("click", analyzeLoadedImages);
  }

  if (classSelect) {
    classSelect.addEventListener("change", updatePromptText);
  }

  if (imageCount) {
    imageCount.addEventListener("input", updatePromptText);
  }

  if (generateBtn) {
    generateBtn.addEventListener("click", async () => {
      await loadGeneratedPreviewImages();
      updateDistributionFromLoadedImages();
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      clearPreview();
      renderDefaultChart();
    });
  }

  if (analyseBtn) {
    analyseBtn.addEventListener("click", openValidationDetailsPage);
  }

  const exportFinalDatasetBtn = document.getElementById("exportFinalDatasetBtn");

  if (exportFinalDatasetBtn) {
    exportFinalDatasetBtn.addEventListener("click", exportFinalGradedDataset);
  }

});