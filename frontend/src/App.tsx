import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

type Region = {
  name: string;
  box: number[][];
  skin_fraction: number;
  valid: boolean;
};

type Person = {
  id: number | string;
  bpm: number | null;
  signal_quality: number | null;
  status: string;
  buffer_samples: number;
  regions: Region[];
};

type Metrics = {
  connected: boolean;
  status: string;
  fps: number | null;
  people: Person[];
  session_seconds?: number;
};

type FrameMessage = {
  metrics: Metrics;
  frame: string | null;
};

type HistoryEntry = {
  key: string;
  time: string;
  bpm: number;
  quality: string;
};

const API = "http://127.0.0.1:8001";
const WS_URL = "ws://127.0.0.1:8001/ws";

function formatTime(seconds = 0) {
  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatNumber(value: unknown, digits = 1, fallback = "—") {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return value.toFixed(digits);
}

function qualityLabel(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "Unavailable";
  if (value >= 0.6) return "High concentration";
  if (value >= 0.25) return "Moderate concentration";
  return "Low concentration";
}

function App() {
  const [metrics, setMetrics] = useState<Metrics>({
    connected: false,
    status: "Waiting for backend",
    fps: null,
    people: [],
  });

  const [frame, setFrame] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [sessionActive, setSessionActive] = useState(false);
  const [personHistories, setPersonHistories] = useState<
    Record<string, HistoryEntry[]>
  >({});

  const socketRef = useRef<WebSocket | null>(null);
  const historyKeysRef = useRef<Record<string, string>>({});
  const reconnectTimerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const connectSocket = useCallback(() => {
    if (stoppedRef.current) return;

    const current = socketRef.current;
    if (
      current?.readyState === WebSocket.OPEN ||
      current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      setBackendOnline(true);
      setMessage("");
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as FrameMessage;

        if (data.metrics) {
          setMetrics(data.metrics);
          setSessionActive(data.metrics.connected);

          for (const person of data.metrics.people ?? []) {
            if (person.bpm == null || !Number.isFinite(person.bpm)) {
              continue;
            }

            const personKey = String(person.id);
            const bpm = Math.round(person.bpm);
            const readingKey = `${bpm}:${person.status}`;

            if (historyKeysRef.current[personKey] === readingKey) {
              continue;
            }

            historyKeysRef.current[personKey] = readingKey;

            const entry: HistoryEntry = {
              key: `${Date.now()}-${personKey}-${readingKey}`,
              time: new Date().toLocaleTimeString(),
              bpm: person.bpm,
              quality: qualityLabel(person.signal_quality),
            };

            setPersonHistories((previous) => ({
              ...previous,
              [personKey]: [
                entry,
                ...(previous[personKey] ?? []),
              ].slice(0, 200),
            }));
          }
        }

        if (data.frame) setFrame(data.frame);
      } catch {
        setMessage("The backend returned an unreadable message.");
      }
    };

    socket.onerror = () => {
      setBackendOnline(false);
    };

    socket.onclose = () => {
      if (socketRef.current === socket) {
        socketRef.current = null;
      }

      setBackendOnline(false);

      if (!stoppedRef.current) {
        reconnectTimerRef.current = window.setTimeout(
          connectSocket,
          2000,
        );
      }
    };
  }, []);

  useEffect(() => {
    stoppedRef.current = false;
    connectSocket();

    return () => {
      stoppedRef.current = true;

      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }

      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connectSocket]);

  const startCamera = async () => {
    setBusy(true);
    setMessage("");

    try {
      const response = await fetch(`${API}/api/start`, {
        method: "POST",
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Unable to start monitoring.");
      }

      setPersonHistories({});
      historyKeysRef.current = {};
      setMessage("Monitoring started. Signal collection is underway.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Unable to start monitoring.",
      );
    } finally {
      setBusy(false);
    }
  };

  const stopCamera = async () => {
    setBusy(true);

    try {
      const response = await fetch(`${API}/api/stop`, {
        method: "POST",
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Unable to stop monitoring.");
      }

      setSessionActive(false);
      setMessage("Monitoring stopped. Session history remains available.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Unable to stop monitoring.",
      );
    } finally {
      setBusy(false);
    }
  };

  const exportCsv = () => {
    window.location.href = `${API}/api/export.csv`;
  };

  const people = Array.isArray(metrics.people) ? metrics.people : [];

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
        <a className="nav-item active" href="#overview">
          <span>◫</span> Overview
        </a>
        <a className="nav-item" href="#monitor">
          <span>◎</span> Live monitor
        </a>
        <a className="nav-item" href="#history">
          <span>↗</span> Session history
        </a>

        <div className="sidebar-bottom">
          <div className="backend-indicator">
            <span className={`status-dot ${backendOnline ? "online" : ""}`} />
            <div>
              <strong>{backendOnline ? "API connected" : "API disconnected"}</strong>
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

          <div className="topbar-right">
            <span className={`connection-pill ${backendOnline ? "connected" : ""}`}>
              <span className="status-dot" />
              {backendOnline ? "Backend online" : "Waiting for backend"}
            </span>
            <span className="date-label">{new Date().toLocaleDateString()}</span>
          </div>
        </header>

        {message && <div className="notice" role="status">{message}</div>}

        <section className="metric-grid" aria-label="Session metrics">
          <article className="metric-card">
            <span className="metric-label">PEOPLE DETECTED</span>
            <strong className="metric-number">{people.length}</strong>
            <span className="metric-foot">Current camera frame</span>
          </article>

          <article className="metric-card">
            <span className="metric-label">PROCESSING FPS</span>
            <strong className="metric-number">{metrics.fps ?? "—"}</strong>
            <span className="metric-foot">Measured processing rate</span>
          </article>

          <article className="metric-card">
            <span className="metric-label">SESSION DURATION</span>
            <strong className="metric-number">
              {formatTime(metrics.session_seconds)}
            </strong>
            <span className="metric-foot">Current capture session</span>
          </article>

          <article className="metric-card">
            <span className="metric-label">PIPELINE STATUS</span>
            <strong className="metric-status">{metrics.status}</strong>
            <span className="metric-foot">Live backend state</span>
          </article>
        </section>

        <section className="monitor-layout" id="monitor">
          <article className="panel camera-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">LIVE INPUT</span>
                <h2>Camera monitor</h2>
              </div>

              <span className={`live-badge ${sessionActive ? "active" : ""}`}>
                <span className="status-dot" />
                {sessionActive ? "LIVE" : "IDLE"}
              </span>
            </div>

            <div className="camera-view">
              {frame ? (
                <img
                  className="camera-image"
                  src={`data:image/jpeg;base64,${frame}`}
                  alt="Live webcam with face and forehead and cheek overlays"
                />
              ) : (
                <div className="camera-placeholder">
                  <div className="camera-symbol">◎</div>
                  <strong>Camera feed will appear here</strong>
                  <p>Start monitoring to see the live image and ROI overlays.</p>
                </div>
              )}
              {frame && (
                <span className="camera-caption">PYTHON VISION PIPELINE</span>
              )}
            </div>

            <div className="camera-controls">
              <button
                className="button primary"
                onClick={startCamera}
                disabled={busy || !backendOnline || sessionActive}
              >
                {busy ? "Please wait…" : "Start monitoring"}
              </button>

              <button
                className="button secondary"
                onClick={stopCamera}
                disabled={busy || !sessionActive}
              >
                Stop
              </button>

              <button className="button secondary" onClick={exportCsv}>
                Export CSV ↓
              </button>
            </div>

            <p className="helper-text">
              ROI colours: green = forehead · amber = left cheek · blue = right cheek.
            </p>
          </article>

          <article className="panel people-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">TRACKING</span>
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
                  <div className="person-card" key={person.id}>
                    <div className="person-top">
                      <div className="person-avatar">P{person.id}</div>
                      <div className="person-title">
                        <strong>Person {person.id}</strong>
                        <small>{person.status}</small>
                      </div>
                      <span className="tracking-tag">TRACKED</span>
                    </div>

                    <div className="bpm-reading">
                      <strong>
                        {person.bpm == null ? "—" : formatNumber(person.bpm, 1)}
                      </strong>
                      <span>BPM</span>
                    </div>

                    <div className="person-stats">
                      <div>
                        <span>Signal quality score</span>
                        <strong>
                          {person.signal_quality == null
                            ? "Unavailable"
                            : formatNumber(person.signal_quality, 3, "Unavailable")}
                        </strong>
                      </div>
                      <div>
                        <span>Valid RGB samples</span>
                        <strong>{person.buffer_samples}</strong>
                      </div>
                      <div>
                        <span>Signal quality label</span>
                        <strong>{qualityLabel(person.signal_quality)}</strong>
                      </div>
                    </div>

                    <div className="roi-status">
                      {(person.regions ?? []).map((region) => (
                        <span
                          className={region.valid ? "roi valid" : "roi"}
                          key={region.name}
                        >
                          {region.name.replace("_", " ")}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="panel-footnote">
              Tracking IDs are temporary and may change if a person leaves the frame.
            </div>
          </article>
        </section>

        <section className="panel history-panel" id="history">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">RECENT OBSERVATIONS</span>
              <h2>Session history by person</h2>
            </div>
            <button className="text-button" onClick={exportCsv}>
              Download CSV ↗
            </button>
          </div>

          {people.length === 0 ? (
            <div className="history-empty">
              Start monitoring to view individual histories. Existing session
              history remains in the dashboard until you start another session
              or refresh the page.
            </div>
          ) : (
            <div className="person-history-grid">
              {people.map((person) => {
                const entries = personHistories[String(person.id)] ?? [];

                return (
                  <article className="person-history-card" key={person.id}>
                    <div className="person-history-heading">
                      <div>
                        <strong>Person {person.id}</strong>
                        <span>{entries.length} observations</span>
                      </div>
                      <span className="history-current-bpm">
                        {person.bpm == null
                          ? "—"
                          : `${formatNumber(person.bpm, 1)} BPM`}
                      </span>
                    </div>

                    <div className="person-history-scroll">
                      {entries.length === 0 ? (
                        <p className="history-empty">
                          No valid BPM estimate recorded yet.
                        </p>
                      ) : (
                        <table>
                          <thead>
                            <tr>
                              <th>Time</th>
                              <th>BPM</th>
                              <th>Quality score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {entries.map((entry) => (
                              <tr key={entry.key}>
                                <td>{entry.time}</td>
                                <td>{formatNumber(entry.bpm, 1)}</td>
                                <td>{entry.quality}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <footer className="disclaimer">
          <span className="disclaimer-mark">i</span>
          <p>
            <strong>Experimental research prototype.</strong> Pulse estimates
            may be inaccurate due to motion, lighting, camera quality, and other
            factors. Signal quality is a rough spectral concentration score,
            not a probability of correctness. Not a medical device; not
            intended for diagnosis, treatment, or emergency decisions.
          </p>
        </footer>
      </section>
    </main>
  );
}

export default App;
