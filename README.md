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
