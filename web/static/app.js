const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileName = document.getElementById("selected-file");
const analyzeButton = document.getElementById("analyze-button");
const statusRegion = document.getElementById("status-region");
const progress = document.getElementById("progress");
const statFileName = document.getElementById("file-name");
const sentenceCount = document.getElementById("sentence-count");
const wordCount = document.getElementById("word-count");
const preview = document.getElementById("preview");
const downloadButton = document.getElementById("download-button");

let selectedFile = null;
let downloadUrl = null;
let outputName = null;
let processing = false;

function setStatus(message = "", type = "") {
  statusRegion.textContent = message;
  statusRegion.className = `status ${type}`.trim();
}

function resetResult() {
  statFileName.textContent = "--";
  sentenceCount.textContent = "--";
  wordCount.textContent = "--";
  preview.textContent = "上传文件后，结果显示在这里。";
  downloadButton.disabled = true;
  downloadUrl = null;
  outputName = null;
}

function chooseFile(file) {
  if (processing) return;
  resetResult();
  setStatus();
  if (!file) {
    selectedFile = null;
    fileName.textContent = "选择或拖入 .txt 文件";
    analyzeButton.disabled = true;
    return;
  }
  if (!file.name.toLowerCase().endsWith(".txt")) {
    selectedFile = null;
    fileName.textContent = "选择或拖入 .txt 文件";
    analyzeButton.disabled = true;
    setStatus("仅支持 .txt 文本文件", "error");
    return;
  }
  selectedFile = file;
  fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
  analyzeButton.disabled = false;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => {
  chooseFile(event.dataTransfer.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile || processing) return;
  processing = true;
  analyzeButton.disabled = true;
  fileInput.disabled = true;
  dropZone.classList.add("disabled");
  dropZone.setAttribute("aria-disabled", "true");
  progress.hidden = false;
  setStatus("正在分析文件...");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    const response = await fetch("/api/segment", { method: "POST", body: formData });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      // Non-JSON error bodies fall back to the HTTP status message.
    }
    if (!response.ok) {
      throw new Error(payload?.error?.message || `请求失败 (${response.status})`);
    }
    if (!payload) {
      throw new Error("处理失败");
    }
    statFileName.textContent = payload.input_name;
    sentenceCount.textContent = payload.sentence_count;
    wordCount.textContent = payload.word_count;
    preview.textContent = payload.preview.join("\n");
    downloadUrl = payload.download_url;
    outputName = payload.output_name;
    downloadButton.disabled = false;
    setStatus("结果文件已生成", "success");
  } catch (error) {
    setStatus(error.message || "处理失败", "error");
  } finally {
    processing = false;
    dropZone.classList.remove("disabled");
    dropZone.setAttribute("aria-disabled", "false");
    progress.hidden = true;
    fileInput.disabled = false;
    analyzeButton.disabled = !selectedFile;
  }
});

downloadButton.addEventListener("click", () => {
  if (!downloadUrl || !outputName) return;
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = outputName;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
});

resetResult();
