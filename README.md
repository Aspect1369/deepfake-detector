# PulseCheck video demo

This project contains a small browser interface, a local Node.js server, and an experimental Python remote-photoplethysmography (rPPG) pulse estimator. The estimator tracks a face with OpenCV, samples green values from forehead and cheek regions, filters those signals in the 0.7–3.3 Hz band, and estimates a pulse peak. It reports BPM and a signal-quality score.

**This is not a deepfake classifier.** A pulse measurement or poor signal cannot tell whether a video is real or fake. The score is not a calibrated confidence or probability. Do not use this demo to accuse a person or make a consequential decision.

## Requirements

- Node.js 22.13 or newer
- npm (included with Node.js)
- Python 3.10 or newer with pip

## Run on Windows

Open PowerShell in this project folder and run:

```powershell
npm install
py -3 -m pip install -r requirements.txt
npm start
```

Open <http://localhost:3000>. Select a well-lit, front-facing video with one visible face, 10–60 seconds long and no larger than 50 MB, then click **Analyze video**. The first upload creates `server-data/pulsecheck.sqlite`.

If the Python launcher cannot find the environment where the detector libraries are installed, set `PYTHON_EXECUTABLE` to the full path of that Python executable before starting Node:

```powershell
$env:PYTHON_EXECUTABLE = "C:\Path\To\python.exe"
npm start
```

To check the JavaScript syntax:

```powershell
npm run check
```

To test the signal processing using generated pulse signals:

```powershell
npm run test:detector
```

## API

- `GET /api/health` reports whether the server and rPPG estimator are configured.
- `POST /api/analyze` accepts one multipart file named `video`.
- `GET /api/analyses` returns up to 50 recent upload records.

The server checks the filename extension, video file signature, and size. It does not preserve the uploaded video. SQLite stores the timestamp, file metadata, status, BPM, signal quality, face-detection rate, pulse strength, and an explanation. The database is local and excluded from Git.

## Interpreting results

`PULSE_ESTIMATED` means a pulse-like peak was found consistently in multiple sampled face regions. `LOW_SIGNAL_QUALITY` or `FACE_NOT_TRACKED` means this clip did not yield a reliable estimate. Camera movement, lighting, compression, skin visibility, and face tracking can all affect the measurement. A pulse estimate does not establish that a video is authentic; a missing or weak pulse does not establish that it is fake.

## Preparing data for a future classifier

The current detector is **not** a real/fake classifier. A separate classifier needs licensed or consented, labeled videos and independent evaluation. Keep video files on your computer; do not commit them to this public repository.

Start with `dataset/manifest.example.csv`. Make a private copy named `dataset/manifest.csv`, place it outside Git if it contains private paths or notes, and use these labels exactly:

- `real`: genuine camera-recorded video.
- `deepfake`: a real source video with a synthetic face manipulation.
- `ai_generated`: video depicting a person generated entirely by AI.

Each `group_id` must link clips that could otherwise leak information across splits: the same participant across recordings, and every manipulated clip with its original source. When groups overlap, assign a shared ID that joins the connected clips. Record generator/tool versions in `method`. Only use data you have permission or a suitable license to analyze.

After you have a folder with the video paths in the CSV, create group-safe 70/15/15 train, validation, and test splits:

```powershell
py -3 prepare_dataset.py --manifest C:\private-data\manifest.csv --dataset-root C:\private-data --output dataset\split_manifest.csv
```

The script validates that paths stay under the dataset root, required labels and fields exist, and each class has at least three distinct groups. It keeps every `group_id` entirely in one split and reports per-split counts. This is data preparation only: it does not train a model or establish accuracy. A later model must be evaluated on untouched people and sources, and performance should be reported per class. With a small dataset, results can be very uncertain; no score guarantees correct predictions on unfamiliar videos.

Run the manifest and split tests with:

```powershell
py -3 -m unittest discover -s tests -v
```
