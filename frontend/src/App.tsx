
import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8001";
const WS_URL = "ws://127.0.0.1:8001/ws";

type Patient = {
  patient_id: string;
  name: string;
  person_label?: string;
  face_photo?: string | null;
  created_at?: string;
};

type Reading = {
  id: number;
  patient_id: string;
  bpm: number | null;
  signal_quality: number | null;
  timestamp: string;
  session_id?: string | null;
};

type Person = {
  id: string | number;
  patient_id?: string | null;
  bpm: number | null;
  signal_quality: number | null;
  status: string;
  buffer_samples: number;
  regions?: { name: string; valid?: boolean }[];
};

type Metrics = {
  connected: boolean;
  status: string;
  fps: number | null;
  people: Person[];
  session_seconds?: number;
};

type WsMessage = {
  metrics?: Metrics;
  frame?: string | null;
};

function formatDate(value?: string | number) {
  if (!value) return "—";
  const date = typeof value === "number"
    ? new Date(value * 1000)
    : new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString();
}

function formatDuration(value = 0) {
  const seconds = Math.max(0, Math.floor(value));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(
    seconds % 60,
  ).padStart(2, "0")}`;
}

function App() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [readings, setReadings] = useState<Reading[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    connected: false,
    status: "Waiting for backend",
    fps: null,
    people: [],
  });
  const [frame, setFrame] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [resetDialog, setResetDialog] = useState(false);
  const [resetBusy, setResetBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);
  const disposedRef = useRef(false);

  const loadPatients = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/patients`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not load patient records.");
      }

      const rows: Patient[] = Array.isArray(data)
        ? data
        : data.patients ?? [];

      setPatients(rows);
      setSelectedId((current) =>
        current && rows.some((p) => p.patient_id === current)
          ? current
          : rows[0]?.patient_id ?? "",
      );
    } catch (error) {
      setNotice(
        error instanceof Error ? error.message : "Could not load patient records.",
      );
    }
  }, []);

  useEffect(() => {
    void loadPatients();
    const timer = window.setInterval(() => void loadPatients(), 3000);
    return () => window.clearInterval(timer);
  }, [loadPatients]);

  useEffect(() => {
    if (!selectedId) {
      setReadings([]);
      return;
    }

    let cancelled = false;

    async function loadReadings() {
      try {
        const response = await fetch(
          `${API}/api/patients/${encodeURIComponent(selectedId)}/readings`,
        );
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || "Could not load patient readings.");
        }

        if (!cancelled) {
          setReadings(Array.isArray(data) ? data : data.readings ?? []);
        }
      } catch (error) {
        if (!cancelled) {
          setReadings([]);
          setNotice(
            error instanceof Error ? error.message : "Could not load readings.",
          );
        }
      }
    }

    void loadReadings();
    const timer = window.setInterval(() => void loadReadings(), 3000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedId]);

  const connectWebSocket = useCallback(() => {
    if (disposedRef.current) return;

    const current = socketRef.current;
    if (
      current?.readyState === WebSocket.OPEN ||
      current?.readyState === WebSocket.CONNECTING
    ) return;

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => setApiOnline(true);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WsMessage;
        if (data.metrics) setMetrics(data.metrics);
        if (data.frame) setFrame(data.frame);
      } catch {
        setNotice("Received an invalid message from the backend.");
      }
    };

    socket.onerror = () => setApiOnline(false);

    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      setApiOnline(false);

      if (!disposedRef.current) {
        reconnectRef.current = window.setTimeout(connectWebSocket, 2000);
      }
    };
  }, []);

  useEffect(() => {
    disposedRef.current = false;
    connectWebSocket();

    return () => {
      disposedRef.current = true;
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current);
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connectWebSocket]);

  const startMonitoring = async () => {
    setBusy(true);
    setNotice("");

    try {
      const response = await fetch(`${API}/api/start`, { method: "POST" });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.detail || data.error || "Could not start camera.");
      }

      setNotice("Monitoring started. Face recognition is initializing.");
      await loadPatients();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not start camera.");
    } finally {
      setBusy(false);
    }
  };

  const stopMonitoring = async () => {
    setBusy(true);

    try {
      const response = await fetch(`${API}/api/stop`, { method: "POST" });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.detail || data.error || "Could not stop camera.");
      }

      setNotice("Monitoring stopped. Saved records remain in the database.");
      await loadPatients();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not stop camera.");
    } finally {
      setBusy(false);
    }
  };

  const resetDatabase = async () => {
    setResetBusy(true);
    setNotice("");

    try {
      const response = await fetch(`${API}/api/patients/clear`, {
        method: "DELETE",
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.detail || data.error || "Database reset failed.");
      }

      setPatients([]);
      setSelectedId("");
      setReadings([]);
      setResetDialog(false);
      setNotice("Database reset successfully. All saved patient data was erased.");
      await loadPatients();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Database reset failed.");
    } finally {
      setResetBusy(false);
    }
  };

  const selectedPatient = patients.find((p) => p.patient_id === selectedId);
  const people = metrics.people ?? [];

  return (
    <main className="pv-app">
      <aside className="sidebar">
        <a className="brand" href="#overview">
          <span className="brand-mark"><span /></span>
          <span>
            <strong>pulsevision</strong>
            <small>CONTACTLESS VITALS LAB</small>
          </span>
        </a>

        <div className="nav-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview">◫ &nbsp; Overview</a>
        <a className="nav-item" href="#monitor">◎ &nbsp; Live monitor</a>
        <a className="nav-item" href="#patients">♙ &nbsp; Patient records</a>
        <a className="nav-item" href="#history">↗ &nbsp; Reading history</a>

        <div className="sidebar-bottom">
          <div className="backend-indicator">
            <span className={`status-dot ${apiOnline ? "online" : ""}`} />
            <div>
              <strong>{apiOnline ? "API connected" : "API disconnected"}</strong>
              <small>127.0.0.1:8001</small>
            </div>
          </div>
          <p className="sidebar-note">
            Research prototype
            <br />
            Not for clinical decisions
          </p>
        </div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <div>
            <div className="eyebrow">RESEARCH / REAL-TIME ANALYSIS</div>
            <h1>Monitoring overview</h1>
            <p>Contactless pulse estimation from facial video.</p>
          </div>
          <span className={`connection-pill ${apiOnline ? "connected" : ""}`}>
            <span className="status-dot" />
            {apiOnline ? "Backend online" : "Waiting for backend"}
          </span>
        </header>

        {notice && <div className="notice" role="status">{notice}</div>}

        <section className="metric-grid">
          <article className="metric-card">
            <span className="metric-label">PEOPLE DETECTED</span>
            <strong className="metric-number">{people.length}</strong>
            <span className="metric-foot">Current camera frame</span>
          </article>
          <article className="metric-card">
            <span className="metric-label">SAVED PATIENTS</span>
            <strong className="metric-number">{patients.length}</strong>
            <span className="metric-foot">Persistent identities</span>
          </article>
          <article className="metric-card">
            <span className="metric-label">PROCESSING FPS</span>
            <strong className="metric-number">{metrics.fps ?? "—"}</strong>
            <span className="metric-foot">Measured processing rate</span>
          </article>
          <article className="metric-card">
            <span className="metric-label">SESSION DURATION</span>
            <strong className="metric-number">{formatDuration(metrics.session_seconds)}</strong>
            <span className="metric-foot">{metrics.status}</span>
          </article>
        </section>

        <section className="monitor-layout" id="monitor">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">LIVE INPUT</span>
                <h2>Camera monitor</h2>
              </div>
              <span className={`live-badge ${metrics.connected ? "active" : ""}`}>
                <span className="status-dot" />
                {metrics.connected ? "LIVE" : "IDLE"}
              </span>
            </div>

            <div className="camera-view">
              {frame ? (
                <img
                  className="camera-image"
                  src={`data:image/jpeg;base64,${frame}`}
                  alt="Live webcam feed with detection overlays"
                />
              ) : (
                <div className="camera-placeholder">
                  <div className="camera-symbol">◎</div>
                  <strong>Camera feed will appear here</strong>
                  <p>Start monitoring to detect faces and collect pulse signals.</p>
                </div>
              )}
            </div>

            <div className="camera-controls">
              <button
                className="button primary"
                onClick={startMonitoring}
                disabled={busy || !apiOnline || metrics.connected}
              >
                {busy ? "Please wait…" : "Start monitoring"}
              </button>
              <button
                className="button secondary"
                onClick={stopMonitoring}
                disabled={busy || !metrics.connected}
              >
                Stop
              </button>
            </div>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">REAL-TIME TRACKING</span>
                <h2>People in frame</h2>
              </div>
              <span className="count-badge">{people.length} detected</span>
            </div>

            {people.length === 0 ? (
              <div className="empty-state">
                <span className="empty-icon">◎</span>
                <strong>No people detected yet</strong>
                <p>Start monitoring and face a well-lit camera.</p>
              </div>
            ) : (
              <div className="people-list">
                {people.map((person) => (
                  <article className="person-card" key={person.id}>
                    <div className="person-top">
                      <div className="person-avatar">
                        {person.patient_id ?? `T${person.id}`}
                      </div>
                      <div className="person-title">
                        <strong>{person.patient_id ?? "Identifying person…"}</strong>
                        <small>{person.status}</small>
                      </div>
                    </div>
                    <div className="bpm-reading">
                      <strong>{person.bpm == null ? "—" : person.bpm.toFixed(1)}</strong>
                      <span>BPM</span>
                    </div>
                    <div className="person-stats">
                      <div>
                        <span>Signal quality</span>
                        <strong>
                          {person.signal_quality == null
                            ? "—"
                            : person.signal_quality.toFixed(3)}
                        </strong>
                      </div>
                      <div>
                        <span>Buffer samples</span>
                        <strong>{person.buffer_samples}</strong>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </article>
        </section>

        <section className="panel patient-records-panel" id="patients">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">PERSISTENT DATABASE</span>
              <h2>Patient records</h2>
              <p className="patient-subtitle">
                Faces, identities and BPM history saved automatically.
              </p>
            </div>
            <div className="patient-header-actions">
              <span className="count-badge">{patients.length} registered</span>
              <button
                className="patient-reset-button"
                onClick={() => setResetDialog(true)}
                disabled={resetBusy || metrics.connected}
                title={
                  metrics.connected
                    ? "Stop monitoring before resetting the database"
                    : "Erase saved patient records"
                }
              >
                Reset database
              </button>
            </div>
          </div>

          {resetDialog && (
            <div className="patient-reset-confirmation" role="alertdialog">
              <strong>Erase all patient data?</strong>
              <p>
                This permanently deletes saved face photos, identities,
                readings and sessions from the database.
              </p>
              <div className="patient-reset-actions">
                <button
                  className="button secondary"
                  onClick={() => setResetDialog(false)}
                  disabled={resetBusy}
                >
                  Cancel
                </button>
                <button
                  className="patient-reset-button"
                  onClick={resetDatabase}
                  disabled={resetBusy}
                >
                  {resetBusy ? "Resetting…" : "Erase all data"}
                </button>
              </div>
            </div>
          )}

          {patients.length === 0 ? (
            <div className="empty-state">
              <span className="empty-icon">♙</span>
              <strong>No patient records yet</strong>
              <p>
                Start monitoring. Detected faces will be matched against
                stored identities, and new identities will be saved automatically.
              </p>
            </div>
          ) : (
            <div className="registered-patients-layout">
              <div className="registered-patients-list">
                {patients.map((patient, index) => (
                  <button
                    type="button"
                    className={`registered-patient-card ${
                      selectedId === patient.patient_id ? "selected" : ""
                    }`}
                    key={patient.patient_id}
                    onClick={() => setSelectedId(patient.patient_id)}
                  >
                    {patient.face_photo ? (
                      <img
                        src={patient.face_photo}
                        alt={`Face for ${patient.person_label || patient.name}`}
                      />
                    ) : (
                      <span className="registered-patient-avatar">
                        P{index + 1}
                      </span>
                    )}
                    <span className="registered-patient-info">
                      <strong>{patient.person_label || patient.name}</strong>
                      <small>ID: {patient.patient_id}</small>
                      <small>Saved: {formatDate(patient.created_at)}</small>
                    </span>
                  </button>
                ))}
              </div>

              <div className="saved-readings-panel">
                {selectedPatient ? (
                  <>
                    <div className="saved-patient-heading">
                      <div>
                        <span className="eyebrow">SELECTED PATIENT</span>
                        <h3>{selectedPatient.person_label || selectedPatient.name}</h3>
                        <p>{selectedPatient.patient_id}</p>
                      </div>
                      {selectedPatient.face_photo && (
                        <img
                          src={selectedPatient.face_photo}
                          alt="Selected patient's face"
                        />
                      )}
                    </div>

                    <h3 className="saved-readings-title">Saved BPM readings</h3>
                    {readings.length === 0 ? (
                      <p className="history-empty">
                        No valid BPM readings have been saved for this patient yet.
                      </p>
                    ) : (
                      <div className="person-history-scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>Timestamp</th>
                              <th>BPM</th>
                              <th>Signal quality</th>
                            </tr>
                          </thead>
                          <tbody>
                            {readings.map((reading, index) => (
                              <tr key={reading.id ?? `${reading.timestamp}-${index}`}>
                                <td>{formatDate(reading.timestamp)}</td>
                                <td>
                                  {reading.bpm == null ? "—" : reading.bpm.toFixed(1)}
                                </td>
                                <td>
                                  {reading.signal_quality == null
                                    ? "—"
                                    : reading.signal_quality.toFixed(3)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="history-empty">Select a patient to view their readings.</p>
                )}
              </div>
            </div>
          )}
        </section>

        <section className="panel history-panel" id="history">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">EXPORT</span>
              <h2>Session observations</h2>
            </div>
            <button
              className="button secondary"
              onClick={() => {
                window.location.href = `${API}/api/export.csv`;
              }}
            >
              Export CSV ↓
            </button>
          </div>
          <p className="history-empty">
            The CSV export contains the current monitoring session. Permanent
            patient readings are shown in the selected patient record.
          </p>
        </section>

        <footer className="disclaimer">
          <span className="disclaimer-mark">i</span>
          <p>
            <strong>Experimental research prototype.</strong>{" "}
            Camera-based pulse estimates may be inaccurate due to movement,
            lighting and camera quality. This is not a medical device and
            must not be used for diagnosis or emergency decisions.
          </p>
        </footer>
      </section>
    </main>
  );
}

export default App;
