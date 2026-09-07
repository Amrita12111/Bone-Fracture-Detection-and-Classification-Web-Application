/* ═══════════════════════════════════════════════════════════════
   OsteoScan AI — app.js
   Handles: detect page logic, health check, drag-and-drop,
            confidence slider, fracture rendering.
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:5000";

/* ── Detect-page element refs (null-safe — only initialised when present) ── */
const dropZone    = document.getElementById("dropZone");
const fileInput   = document.getElementById("fileInput");
const confSlider  = document.getElementById("confSlider");
const confVal     = document.getElementById("confVal");
const detectBtn   = document.getElementById("detectBtn");
const btnText     = document.getElementById("btnText");

const previewPanel    = document.getElementById("previewPanel");
const previewImg      = document.getElementById("previewImg");
const resultsSection  = document.getElementById("resultsSection");
const noFracturePanel = document.getElementById("noFracturePanel");
const resultImg       = document.getElementById("resultImg");
const inferenceTime   = document.getElementById("inferenceTime");
const downloadBtn     = document.getElementById("downloadBtn");
const summaryCard     = document.getElementById("summaryCard");
const detectionsList  = document.getElementById("detectionsList");

let selectedFile = null;

/* ── API Health Check ── */
async function checkHealth() {
  const dot = document.getElementById("statusDot");
  if (!dot) return;
  try {
    const r = await fetch(`${API_BASE}/health`);
    dot.style.background = r.ok ? "#57f1db" : "#ef4444";
    dot.style.boxShadow  = r.ok
      ? "0 0 8px rgba(87,241,219,0.7)"
      : "0 0 8px rgba(239,68,68,0.5)";
  } catch {
    dot.style.background = "#ef4444";
    dot.style.boxShadow  = "0 0 8px rgba(239,68,68,0.5)";
  }
}
checkHealth();

/* ── Drag & Drop (only wire if elements exist) ── */
if (dropZone) {
  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("active");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("active"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("active");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  dropZone.addEventListener("click", () => fileInput && fileInput.click());
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
}

/* ── Confidence Slider ── */
if (confSlider && confVal) {
  confSlider.addEventListener("input", () => {
    confVal.textContent = confSlider.value + "%";
  });
}

/* ── File Validation & Preview ── */
function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (JPEG, PNG, DICOM-converted, etc.).");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    alert("File too large. Maximum allowed size is 16 MB.");
    return;
  }

  selectedFile = file;
  const url = URL.createObjectURL(file);

  if (previewImg)  previewImg.src = url;
  if (previewPanel) previewPanel.style.display = "block";
  if (resultsSection)  resultsSection.style.display  = "none";
  if (noFracturePanel) noFracturePanel.style.display = "none";
  if (detectBtn)   detectBtn.disabled = false;

  const dropTitle = dropZone && dropZone.querySelector(".drop-title");
  if (dropTitle) dropTitle.textContent = file.name;
}

/* ── Detect Button ── */
if (detectBtn) {
  detectBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    detectBtn.disabled = true;
    if (btnText) {
      btnText.innerHTML = `<span class="spinner"></span> Analyzing…`;
    }

    const formData = new FormData();
    formData.append("image",      selectedFile);
    formData.append("confidence", confSlider ? confSlider.value / 100 : 0.5);

    try {
      const res  = await fetch(`${API_BASE}/detect`, { method: "POST", body: formData });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Detection failed");
      renderResults(data);
    } catch (err) {
      alert("Detection error: " + err.message);
    } finally {
      detectBtn.disabled = false;
      if (btnText) btnText.textContent = "Detect Fractures";
    }
  });
}

/* ── Render Detection Results ── */
function renderResults(data) {
  // No fracture path
  if (data.no_fracture) {
    if (noFracturePanel) noFracturePanel.style.display = "block";
    if (resultsSection)  resultsSection.style.display  = "none";
    return;
  }

  if (noFracturePanel) noFracturePanel.style.display = "none";
  if (resultsSection)  resultsSection.style.display  = "grid";

  if (resultImg)     resultImg.src = data.annotated_image;
  if (inferenceTime) inferenceTime.textContent = `⏱ ${data.inference_time_ms} ms`;

  // Download button
  if (downloadBtn) {
    downloadBtn.onclick = () => {
      const a = document.createElement("a");
      a.href     = data.annotated_image;
      a.download = `fracture_result_${Date.now()}.jpg`;
      a.click();
    };
  }

  // Severity colour map
  const severityColors = {
    Critical: "#dc2626",
    High:     "#ef4444",
    Medium:   "#f97316",
    Low:      "#f59e0b",
    None:     "#57f1db"
  };
  const sColor = severityColors[data.highest_severity] || "#8b949e";

  // Summary card
  if (summaryCard) {
    summaryCard.innerHTML = `
      <div class="summary-label">Overall Assessment</div>
      <div class="summary-value" style="color:${sColor}">${data.highest_severity} Severity</div>
      <div style="margin-top:8px;font-size:12px;color:var(--muted)">
        ${data.total_detected} fracture${data.total_detected !== 1 ? "s" : ""} detected
      </div>
    `;
  }

  // Detection cards
  if (detectionsList) {
    detectionsList.innerHTML = "";
    data.detections.forEach((det, i) => {
      const card = document.createElement("div");
      card.className = "detection-card";
      card.style.animationDelay = `${i * 0.08}s`;
      card.innerHTML = `
        <div class="det-header">
          <span class="det-class">${det.class.replace(/_/g, " ")}</span>
          <span class="det-conf">${det.confidence}%</span>
        </div>
        <span class="det-severity" style="background:${det.color}22;color:${det.color};border:1px solid ${det.color}44">
          ${det.severity} Severity
        </span>
        <p class="det-advice">${det.advice}</p>
        <div class="conf-bar-wrap">
          <div class="conf-bar" style="width:${det.confidence}%;background:${det.color}"></div>
        </div>
      `;
      detectionsList.appendChild(card);
    });
  }
}