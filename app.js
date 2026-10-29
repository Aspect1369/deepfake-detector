"use strict";

const MAX_FILE_SIZE = 50 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["mp4", "mov", "webm"]);

const fileInput = document.querySelector("#video-input");
const dropZone = document.querySelector("#drop-zone");
const message = document.querySelector("#file-message");
const previewWrap = document.querySelector("#preview-wrap");
const videoPreview = document.querySelector("#video-preview");
const previewError = document.querySelector("#preview-error");
const fileName = document.querySelector("#file-name");
const removeButton = document.querySelector("#remove-video");
const analyzeButton = document.querySelector("#analyze-button");
const resultCard = document.querySelector("#result-card");
const pulseGraph = document.querySelector("#pulse-graph");

let selectedVideo = null;
let previewUrl = null;
let activeRequestController = null;

function showMessage(text) {
	message.textContent = text;
}

function clearSelection() {
	if (activeRequestController) {
		activeRequestController.abort();
		activeRequestController = null;
	}
	setLoading(false);

	if (previewUrl) {
		URL.revokeObjectURL(previewUrl);
		previewUrl = null;
	}

	selectedVideo = null;
	videoPreview.removeAttribute("src");
	videoPreview.load();
	previewError.hidden = true;
	fileInput.value = "";
	fileName.textContent = "";
	previewWrap.hidden = true;
	analyzeButton.disabled = true;
	resultCard.hidden = true;
}

function validateVideo(file) {
	if (!file) {
		return "Choose a video file first.";
	}

	const extension = file.name.split(".").pop().toLowerCase();
	if (!ALLOWED_EXTENSIONS.has(extension)) {
		return "Choose an MP4, MOV or WebM video.";
	}

	if (file.size === 0) {
		return "That file is empty. Choose a different video.";
	}

	if (file.size > MAX_FILE_SIZE) {
		return "That video is larger than 50 MB. Choose a smaller file.";
	}

	return "";
}

function useVideo(file) {
	clearSelection();
	const error = validateVideo(file);
	if (error) {
		showMessage(error);
		return;
	}

	showMessage("");
	selectedVideo = file;
	previewUrl = URL.createObjectURL(file);
	videoPreview.src = previewUrl;
	fileName.textContent = `${file.name} - ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
	previewWrap.hidden = false;
	analyzeButton.disabled = false;
}

function setLoading(isLoading) {
	analyzeButton.classList.toggle("is-loading", isLoading);
	analyzeButton.disabled = isLoading || !selectedVideo;
	analyzeButton.querySelector(".button-label").textContent = isLoading ? "Sending to server..." : "Analyze video";
	analyzeButton.setAttribute("aria-busy", String(isLoading));
}

function showResult(result) {
	document.querySelector("#result-summary").textContent = `Upload record #${result.recordId} · ${new Date(result.createdAt).toLocaleString()}`;
	document.querySelector("#result-message").textContent = result.message;
	document.querySelector("#verdict").textContent = result.verdict;
	document.querySelector("#confidence").textContent = `${result.signalQuality}%`;
	document.querySelector("#bpm").textContent = result.bpm === null ? "Not available" : result.bpm;
	document.querySelector("#result-footnote").textContent = `Face tracked in ${Math.round(result.faceDetectionRate * 100)}% of sampled frames. Signal quality is not deepfake confidence. This pulse measurement cannot determine authenticity.`;
	pulseGraph.hidden = !Array.isArray(result.pulseSignal) || result.pulseSignal.length < 2;
	if (!pulseGraph.hidden) {
		const points = result.pulseSignal.map((value, index) => {
			const x = index * 500 / (result.pulseSignal.length - 1);
			const y = 54 - Math.max(-1, Math.min(1, Number(value))) * 38;
			return `${x.toFixed(1)},${y.toFixed(1)}`;
		});
		const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 108">
			<rect width="500" height="108" fill="#fbfcff"/>
			<path d="M0 27.5H500M0 54H500M0 80.5H500" stroke="#edf0f6" stroke-width="1"/>
			<polyline points="${points.join(" ")}" fill="none" stroke="#5369df" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
		</svg>`;
		pulseGraph.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
	}
	resultCard.hidden = false;
	resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

fileInput.addEventListener("change", () => {
	useVideo(fileInput.files[0]);
});

for (const eventName of ["dragenter", "dragover"]) {
	dropZone.addEventListener(eventName, (event) => {
		event.preventDefault();
		dropZone.classList.add("is-dragging");
	});
}

for (const eventName of ["dragleave", "drop"]) {
	dropZone.addEventListener(eventName, (event) => {
		event.preventDefault();
		dropZone.classList.remove("is-dragging");
	});
}

dropZone.addEventListener("drop", (event) => {
	useVideo(event.dataTransfer.files[0]);
});

videoPreview.addEventListener("error", () => {
	previewError.hidden = false;
});

removeButton.addEventListener("click", () => {
	clearSelection();
	showMessage("Video removed. Choose another file when you are ready.");
});

analyzeButton.addEventListener("click", async () => {
	if (!selectedVideo || analyzeButton.disabled) {
		return;
	}

	resultCard.hidden = true;
	showMessage("");
	const requestController = new AbortController();
	activeRequestController = requestController;
	setLoading(true);

	try {
		const formData = new FormData();
		formData.append("video", selectedVideo);
		const response = await fetch("/api/analyze", {
			method: "POST",
			body: formData,
			signal: requestController.signal
		});
		const result = await response.json();
		if (!response.ok) {
			throw new Error(result.error || "The server could not process this video.");
		}
		if (typeof result.recordId !== "number" || !result.createdAt || !result.verdict || !result.message) {
			throw new Error("The server returned an unexpected response.");
		}
		showResult(result);
	} catch (error) {
		if (!(error instanceof Error && error.name === "AbortError")) {
			showMessage(error instanceof Error ? error.message : "Could not connect to the local server.");
		}
	} finally {
		if (activeRequestController === requestController) {
			activeRequestController = null;
			setLoading(false);
		}
	}
});

window.addEventListener("beforeunload", () => {
	if (previewUrl) {
		URL.revokeObjectURL(previewUrl);
	}
});
