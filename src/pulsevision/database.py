
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "pulsevision.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_photo TEXT,
                face_embedding TEXT,
                person_label TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                FOREIGN KEY(patient_id)
                    REFERENCES patients(patient_id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                session_id TEXT,
                bpm REAL,
                confidence REAL,
                signal_quality REAL,
                timestamp TEXT NOT NULL,
                raw_data TEXT,
                FOREIGN KEY(patient_id)
                    REFERENCES patients(patient_id) ON DELETE CASCADE,
                FOREIGN KEY(session_id)
                    REFERENCES sessions(session_id) ON DELETE SET NULL
            )
        """)

        # Migrate older databases without dropping existing records.
        migrations = {
            "patients": {
                "face_photo": "TEXT",
                "face_embedding": "TEXT",
                "person_label": "TEXT",
                "created_at": "TEXT",
            },
            "sessions": {
                "ended_at": "TEXT",
            },
            "readings": {
                "confidence": "REAL",
                "signal_quality": "REAL",
                "timestamp": "TEXT",
                "raw_data": "TEXT",
            },
        }

        for table, columns in migrations.items():
            existing = {
                row["name"]
                for row in conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }

            for column, sql_type in columns.items():
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                    )

        # Backfill metadata on legacy rows.
        rows = conn.execute(
            "SELECT patient_id, name FROM patients ORDER BY rowid"
        ).fetchall()

        for index, row in enumerate(rows, start=1):
            conn.execute(
                """
                UPDATE patients
                SET person_label = COALESCE(person_label, ?),
                    created_at = COALESCE(created_at, ?)
                WHERE patient_id = ?
                """,
                (f"Person {index}", now_iso(), row["patient_id"]),
            )

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_readings_patient_time
            ON readings(patient_id, timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_patient_time
            ON sessions(patient_id, started_at DESC)
        """)


def get_all_patients() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT patient_id, name, face_photo, face_embedding,
                   person_label, created_at
            FROM patients
            ORDER BY rowid
        """).fetchall()
    return [dict(row) for row in rows]


def get_patient(patient_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
    return dict(row) if row else None


def create_automatic_patient(
    face_photo: str | None = None,
    face_embedding: Any = None,
) -> dict[str, Any]:
    """Create the next Person N record atomically."""
    init_db()

    embedding_text = (
        json.dumps(face_embedding)
        if face_embedding is not None
        else None
    )

    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")

        # Count existing records under the write lock so concurrent
        # creations cannot receive the same label.
        number = conn.execute(
            "SELECT COUNT(*) AS n FROM patients"
        ).fetchone()["n"] + 1

        patient_id = f"PV-{uuid.uuid4().hex[:8].upper()}"
        label = f"Person {number}"
        created_at = now_iso()

        conn.execute(
            """
            INSERT INTO patients (
                patient_id, name, face_photo, face_embedding,
                person_label, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                label,
                face_photo,
                embedding_text,
                label,
                created_at,
            ),
        )

        conn.commit()

    return get_patient(patient_id) or {}


def create_patient(
    name: str,
    patient_id: str | None = None,
    face_photo: str | None = None,
    face_embedding: Any = None,
) -> dict[str, Any]:
    """Compatibility method for older API code."""
    init_db()

    name = (name or "").strip()
    if not name:
        raise ValueError("Patient name is required.")

    patient_id = (
        (patient_id or "").strip()
        or f"PV-{uuid.uuid4().hex[:8].upper()}"
    )

    with get_connection() as conn:
        number = conn.execute(
            "SELECT COUNT(*) AS n FROM patients"
        ).fetchone()["n"] + 1

        label = f"Person {number}"
        conn.execute(
            """
            INSERT INTO patients (
                patient_id, name, face_photo, face_embedding,
                person_label, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                name,
                face_photo,
                json.dumps(face_embedding)
                if face_embedding is not None else None,
                label,
                now_iso(),
            ),
        )

    return get_patient(patient_id) or {}


def update_patient_photo(
    patient_id: str,
    face_photo: str | None,
) -> bool:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE patients SET face_photo = ? WHERE patient_id = ?",
            (face_photo, patient_id),
        )
        return cur.rowcount > 0


def update_face_embedding(
    patient_id: str,
    face_embedding: Any,
) -> bool:
    init_db()
    value = (
        json.dumps(face_embedding)
        if face_embedding is not None else None
    )
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE patients SET face_embedding = ? WHERE patient_id = ?",
            (value, patient_id),
        )
        return cur.rowcount > 0


def get_face_embedding(patient_id: str) -> Any:
    patient = get_patient(patient_id)
    if not patient or not patient.get("face_embedding"):
        return None

    try:
        return json.loads(patient["face_embedding"])
    except (TypeError, json.JSONDecodeError):
        return None


def create_session(patient_id: str) -> str:
    init_db()
    session_id = str(uuid.uuid4())

    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()

        if not exists:
            raise ValueError("Patient does not exist.")

        conn.execute(
            """
            INSERT INTO sessions (session_id, patient_id, started_at)
            VALUES (?, ?, ?)
            """,
            (session_id, patient_id, now_iso()),
        )

    return session_id


def end_session(session_id: str) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (now_iso(), session_id),
        )


def get_patient_sessions(patient_id: str) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE patient_id = ?
            ORDER BY started_at DESC
            """,
            (patient_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_reading(
    patient_id: str,
    heart_rate: float | None = None,
    confidence: float | None = None,
    signal_quality: float | None = None,
    session_id: str | None = None,
    raw_data: Any = None,
    **extra: Any,
) -> int:
    init_db()

    bpm = heart_rate if heart_rate is not None else extra.get("bpm")
    timestamp = extra.get("timestamp") or now_iso()

    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()

        if not exists:
            raise ValueError(f"Unknown patient ID: {patient_id}")

        cur = conn.execute(
            """
            INSERT INTO readings (
                patient_id, session_id, bpm, confidence,
                signal_quality, timestamp, raw_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                session_id,
                bpm,
                confidence,
                signal_quality,
                timestamp,
                json.dumps(raw_data) if raw_data is not None else None,
            ),
        )
        return int(cur.lastrowid)


def get_patient_readings(
    patient_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 5000))

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, patient_id, session_id, bpm, confidence,
                   signal_quality, timestamp, raw_data
            FROM readings
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def get_latest_reading(patient_id: str) -> dict[str, Any] | None:
    rows = get_patient_readings(patient_id, limit=1)
    return rows[0] if rows else None


def get_reading_count(patient_id: str | None = None) -> int:
    init_db()
    with get_connection() as conn:
        if patient_id is None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM readings"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM readings WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
    return int(row["n"])


def get_all_readings() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM readings ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def clear_all_data() -> dict[str, Any]:
    """Clear PulseVision records while keeping the schema intact."""
    init_db()

    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM readings")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM patients")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name = 'readings'"
        )
        conn.commit()

    return {
        "ok": True,
        "tables_cleared": ["readings", "sessions", "patients"],
    }


init_db()
