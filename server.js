"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const fsSync = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const express = require("express");
const multer = require("multer");
const Database = require("better-sqlite3");

const APP_ROOT = __dirname;
const DATABASE_DIRECTORY = path.join(APP_ROOT, "server-data");
const DATABASE_PATH = path.join(DATABASE_DIRECTORY, "pulsecheck.sqlite");
const MAX_FILE_SIZE = 50 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([".mp4", ".mov", ".webm"]);
const PORT = Number(process.env.PORT) || 3000;
const PYTHON_EXECUTABLE = process.env.PYTHON_EXECUTABLE ||
	(process.platform === "win32" ? "py" : "python3");
const DETECTOR_SCRIPT = path.join(APP_ROOT, "detector", "analyze_video.py");

const app = express();
const upload = multer({
	storage: multer.memoryStorage(),
	limits: { fileSize: MAX_FILE_SIZE, files: 1 },
	fileFilter: (_request, file, callback) => {
		if (!ALLOWED_EXTENSIONS.has(path.extname(file.originalname).toLowerCase())) {
			callback(new Error("Choose an MP4, MOV or WebM video."));
			return;
		}
		callback(null, true);
	}
});

fsSync.mkdirSync(DATABASE_DIRECTORY, { recursive: true });

const database = new Database(DATABASE_PATH);
database.pragma("journal_mode = WAL");
database.exec(`
	CREATE TABLE IF NOT EXISTS analyses (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		created_at TEXT NOT NULL,
		file_extension TEXT NOT NULL,
		file_size_bytes INTEGER NOT NULL,
		status TEXT NOT NULL,
		verdict TEXT NOT NULL,
		confidence REAL,
		bpm REAL,
		signal_quality REAL,
		duration_seconds REAL,
		face_detection_rate REAL,
		pulse_strength REAL,
		message TEXT
	)
`);

const analysisColumns = new Set(database.pragma("table_info(analyses)").map((column) => column.name));
for (const [column, type] of [
	["signal_quality", "REAL"],
	["duration_seconds", "REAL"],
	["face_detection_rate", "REAL"],
	["pulse_strength", "REAL"],
	["message", "TEXT"]
]) {
	if (!analysisColumns.has(column)) {
		database.exec(`ALTER TABLE analyses ADD COLUMN ${column} ${type}`);
	}
}

const saveAnalysis = database.prepare(`
	INSERT INTO analyses (
		created_at, file_extension, file_size_bytes, status, verdict, confidence, bpm,
		signal_quality, duration_seconds, face_detection_rate, pulse_strength, message
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

function runDetector(videoPath) {
	return new Promise((resolve, reject) => {
		const detectorArguments = process.env.PYTHON_EXECUTABLE
			? [DETECTOR_SCRIPT, videoPath]
			: process.platform === "win32"
				? ["-3", DETECTOR_SCRIPT, videoPath]
				: [DETECTOR_SCRIPT, videoPath];
		const detector = spawn(PYTHON_EXECUTABLE, detectorArguments, { windowsHide: true });
		const stdout = [];
		const stderr = [];
		let stdoutLength = 0;
		let timedOut = false;
		const timeout = setTimeout(() => {
			timedOut = true;
			detector.kill();
		}, 120_000);

		detector.stdout.on("data", (chunk) => {
			stdoutLength += chunk.length;
			if (stdoutLength > 1024 * 1024) {
				detector.kill();
				return;
			}
			stdout.push(chunk);
		});
		detector.stderr.on("data", (chunk) => {
			if (Buffer.concat(stderr).length < 64 * 1024) {
				stderr.push(chunk);
			}
		});
		detector.on("error", (error) => {
			clearTimeout(timeout);
			reject(new Error(`Could not start Python detector (${PYTHON_EXECUTABLE}): ${error.message}`));
		});
		detector.on("close", (code) => {
			clearTimeout(timeout);
			if (timedOut) {
				reject(new Error("Video analysis took too long. Try a shorter or smaller video."));
				return;
			}
			if (stdoutLength > 1024 * 1024) {
				reject(new Error("The detector response exceeded its size limit."));
				return;
			}
			const output = Buffer.concat(stdout).toString("utf8").trim();
			if (code !== 0) {
				const detail = Buffer.concat(stderr).toString("utf8").trim() || output;
				reject(new Error(detail || `Detector exited with code ${code}.`));
				return;
			}
			try {
				const result = JSON.parse(output);
				if (result.error) {
					reject(new Error(result.error));
					return;
				}
				resolve(result);
			} catch (error) {
				reject(new Error(`The detector returned invalid JSON: ${error.message}`));
			}
		});
	});
}

function hasExpectedSignature(buffer, extension) {
	if (extension === ".webm") {
		return buffer.length >= 4 &&
			buffer[0] === 0x1a &&
			buffer[1] === 0x45 &&
			buffer[2] === 0xdf &&
			buffer[3] === 0xa3;
	}

	return buffer.length >= 12 && buffer.toString("ascii", 4, 8) === "ftyp";
}

app.get("/", (_request, response) => {
	response.sendFile(path.join(APP_ROOT, "index.html"));
});

app.get("/styles.css", (_request, response) => {
	response.sendFile(path.join(APP_ROOT, "styles.css"));
});

app.get("/app.js", (_request, response) => {
	response.sendFile(path.join(APP_ROOT, "app.js"));
});

app.get("/api/health", (_request, response) => {
	response.json({ status: "ok", detector: "opencv_rppg" });
});

app.get("/api/analyses", (_request, response) => {
	const records = database.prepare(`
		SELECT id, created_at AS createdAt, file_extension AS fileExtension,
			file_size_bytes AS fileSizeBytes, status, verdict, bpm,
			signal_quality AS signalQuality, duration_seconds AS durationSeconds,
			face_detection_rate AS faceDetectionRate, pulse_strength AS pulseStrength,
			message
		FROM analyses
		ORDER BY id DESC
		LIMIT 50
	`).all();
	response.json({ analyses: records });
});

app.post("/api/analyze", (request, response, next) => {
	upload.single("video")(request, response, async (uploadError) => {
		if (uploadError) {
			next(uploadError);
			return;
		}

		if (!request.file) {
			response.status(400).json({ error: "Choose a video file to upload." });
			return;
		}

		const extension = path.extname(request.file.originalname).toLowerCase();
		if (!hasExpectedSignature(request.file.buffer, extension)) {
			response.status(400).json({ error: "The file contents do not match an MP4, MOV or WebM video." });
			return;
		}

		let temporaryPath;
		try {
			temporaryPath = path.join(os.tmpdir(), `pulsecheck-${crypto.randomUUID()}${extension}`);
			await fs.writeFile(temporaryPath, request.file.buffer, { flag: "wx" });

			const result = await runDetector(temporaryPath);
			const createdAt = new Date().toISOString();
			const saved = saveAnalysis.run(
				createdAt,
				extension,
				request.file.size,
				result.status,
				result.verdict,
				null,
				result.bpm,
				result.signalQuality,
				result.durationSeconds,
				result.faceDetectionRate,
				result.pulseStrength,
				result.message
			);

			response.status(200).json({
				recordId: Number(saved.lastInsertRowid),
				createdAt,
				...result,
			});
		} catch (error) {
			next(error);
		} finally {
			if (temporaryPath) {
				try {
					await fs.unlink(temporaryPath);
				} catch (error) {
					if (error.code !== "ENOENT") {
						console.error("Could not delete the temporary upload:", error);
					}
				}
			}
		}
	});
});

app.use((error, _request, response, _next) => {
	if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
		response.status(413).json({ error: "That video is larger than 50 MB. Choose a smaller file." });
		return;
	}
	if (error instanceof multer.MulterError && error.code === "LIMIT_UNEXPECTED_FILE") {
		response.status(400).json({ error: "Upload one video using the video field." });
		return;
	}
	if (error.message === "Choose an MP4, MOV or WebM video.") {
		response.status(400).json({ error: error.message });
		return;
	}

	console.error("Request failed:", error);
	response.status(500).json({ error: "The server could not process the request. Check the server console." });
});

const server = app.listen(PORT, () => {
	console.log(`PulseCheck is running at http://localhost:${PORT}`);
	console.log(`SQLite database: ${DATABASE_PATH}`);
});

function closeServer() {
	server.close(() => {
		database.close();
		process.exit(0);
	});
}

process.on("SIGINT", closeServer);
process.on("SIGTERM", closeServer);
