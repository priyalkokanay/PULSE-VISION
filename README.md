# PulseVision — Contactless Vitals Lab

PulseVision is a webcam-based research project that explores contactless heart-rate estimation using remote photoplethysmography (rPPG). It combines real-time video processing, facial-region analysis, pulse-signal estimation, a web dashboard, and automatic storage of patient records and BPM readings.

**Current milestone: Version 1 complete.** The core prototype is implemented, with further features planned for future versions.

> **Research disclaimer:** PulseVision is an experimental prototype, not a medical device. Its estimates may be inaccurate due to movement, lighting, camera quality, and other factors. It must not be used for medical diagnosis or emergency decisions.

## Version 1 — Completed

### Core monitoring
- Webcam-based face detection and tracking.
- Facial-region extraction for rPPG signal processing.
- Experimental heart-rate estimation from facial video.
- Live camera feed with detection overlays.
- Real-time monitoring status and processing metrics.
- React-based dashboard connected to a Python backend.

### Automatic patient database
- Automatic creation of patient records from camera detections.
- Sequential patient labels such as `Person 1`, `Person 2`, and `Person 3`.
- Automatic face-photo capture and local storage.
- SQLite persistence for patient records, sessions, and BPM readings.
- Automatic storage of valid BPM estimates against patient IDs.
- Patient reading history and latest-reading display.
- CSV export of observations from the current monitoring session.
- Database reset functionality.

### Application infrastructure
- React and TypeScript frontend.
- FastAPI backend.
- WebSocket communication for live monitoring updates.
- Local SQLite database.
- Separate frontend and backend development servers.

## Technology stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, CSS |
| Backend | Python, FastAPI, Uvicorn |
| Computer vision | OpenCV, NumPy |
| Signal processing | rPPG, POS-based pulse estimation where configured |
| Data storage | SQLite |
| Live communication | WebSockets |

## Current limitations

- Persistent recognition of returning people across separate camera sessions is not yet implemented in the current Buffalo-free build.
- Live tracking IDs can change between sessions, potentially creating duplicate patient records.
- BPM estimates depend on lighting, motion, camera quality, and signal quality.
- The prototype is intended for research and demonstration rather than clinical use.

## Roadmap — Future versions

PulseVision is designed to evolve beyond its initial prototype. Potential improvements include:

### Version 2 — Identity and signal reliability
- Persistent patient recognition across sessions.
- Improved duplicate-record prevention.
- More robust face tracking under movement and changing lighting.
- Signal-quality scoring and confidence-based reading storage.
- Improved handling of invalid or unstable BPM estimates.
- Better monitoring of missing frames and camera failures.

### Version 3 — Advanced monitoring
- Additional physiological estimates where technically feasible and validated.
- Historical trends and interactive charts.
- Session comparisons and patient-specific analytics.
- Configurable measurement intervals and signal-quality thresholds.
- Improved data export and reporting.

### Version 4 — Hardware integration
- ESP32-CAM integration for image capture.
- TFT display for local status and measurements.
- Lightweight device-to-server communication.
- Hardware-oriented performance and latency optimization.
- Evaluation of embedded processing constraints.

### Future development ideas
- Optional database backup and restore.
- User authentication and access controls.
- Improved privacy controls and configurable data retention.
- Automated testing and deployment workflows.
- Benchmarking against reference measurements under controlled conditions.

*The roadmap is provisional. Features will be prioritized based on feasibility, performance, privacy, and testing results.*

## How PulseVision works

1. The user starts webcam monitoring.
2. The backend captures video frames and tracks detected faces.
3. Facial regions are processed to collect colour variations over time.
4. The rPPG pipeline attempts to extract a pulse signal.
5. The estimator produces an experimental BPM reading when sufficient usable data is available.
6. Patient records, face photos, sessions, and valid readings are stored locally.
7. The dashboard displays the camera feed, patient records, and reading history.

The current Buffalo-free implementation associates records with live tracking IDs. Persistent cross-session identity recognition remains a future improvement.

## Run locally

### Requirements
- Python and the project's backend dependencies.
- Node.js and npm.
- A compatible webcam.
- A modern web browser.

### Start the backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend:app --host 127.0.0.1 --port 8001
```

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL shown by Vite, usually `http://localhost:5173/`.

### Backend endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Monitoring status |
| POST | `/api/start` | Start monitoring |
| POST | `/api/stop` | Stop monitoring |
| GET | `/api/patients` | List patient records |
| GET | `/api/patients/{patient_id}/readings` | Read patient history |
| GET | `/api/patients/{patient_id}/sessions` | Read patient sessions |
| DELETE | `/api/patients/clear` | Reset saved records |
| GET | `/api/export.csv` | Export current-session observations |
| WebSocket | `/ws` | Stream camera frames and live metrics |

Interactive API documentation is available at `http://127.0.0.1:8001/docs`.

## Data and privacy

PulseVision stores data locally in `pulsevision.db`. Face photos and measurement records can be sensitive, so use informed consent, restrict access, and avoid committing real patient data to version control.

The database reset operation permanently deletes saved patient records, photos, sessions, and readings. Back up important data before resetting.

## Project status

**Version 1: Core prototype complete.**

The next development phase focuses on persistent identity matching, improved signal reliability, stronger data management, and future hardware integration. Each new version will build on the existing camera, signal-processing, database, and dashboard foundations.
