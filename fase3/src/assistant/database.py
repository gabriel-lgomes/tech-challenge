"""
Banco de dados SQLite com prontuários e registros de pacientes.
Permite ao assistente consultar informações estruturadas em tempo real.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = "data/hospital.db"


def init_db(db_path: str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id  TEXT PRIMARY KEY,
            age         INTEGER,
            sex         TEXT,
            diagnosis   TEXT,
            ward        TEXT,
            admission_date TEXT
        );

        CREATE TABLE IF NOT EXISTS medications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT,
            medication  TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS pending_exams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT,
            exam        TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT,
            alert_text  TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
    """)
    conn.commit()
    conn.close()


def seed_from_json(json_path: str = "data/synthetic/patients.json", db_path: str = DB_PATH) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        patients = json.load(f)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for p in patients:
        cur.execute(
            "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?)",
            (p["patient_id"], p["age"], p["sex"], p["diagnosis"], p["ward"], p["admission_date"]),
        )
        for med in p.get("medications", []):
            cur.execute("INSERT INTO medications (patient_id, medication) VALUES (?,?)", (p["patient_id"], med))
        for exam in p.get("pending_exams", []):
            cur.execute("INSERT INTO pending_exams (patient_id, exam) VALUES (?,?)", (p["patient_id"], exam))
        for alert in p.get("alerts", []):
            cur.execute("INSERT INTO alerts (patient_id, alert_text) VALUES (?,?)", (p["patient_id"], alert))

    conn.commit()
    conn.close()
    print(f"Banco populado com {len(patients)} pacientes.")


def get_patient(patient_id: str, db_path: str = DB_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    patient = dict(row)

    cur.execute("SELECT medication FROM medications WHERE patient_id = ?", (patient_id,))
    patient["medications"] = [r["medication"] for r in cur.fetchall()]

    cur.execute("SELECT exam FROM pending_exams WHERE patient_id = ?", (patient_id,))
    patient["pending_exams"] = [r["exam"] for r in cur.fetchall()]

    cur.execute("SELECT alert_text FROM alerts WHERE patient_id = ?", (patient_id,))
    patient["alerts"] = [r["alert_text"] for r in cur.fetchall()]

    conn.close()
    return patient


def get_pending_exams(patient_id: str, db_path: str = DB_PATH) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT exam FROM pending_exams WHERE patient_id = ?", (patient_id,))
    exams = [row[0] for row in cur.fetchall()]
    conn.close()
    return exams


def add_alert(patient_id: str, alert_text: str, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO alerts (patient_id, alert_text) VALUES (?,?)", (patient_id, alert_text))
    conn.commit()
    conn.close()
