
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "artifacts/inspection_log.db"


def initialize_log():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspected_at TEXT NOT NULL,
                shift TEXT NOT NULL,
                image_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)


def save_inspection(
    image_name, predicted_class, confidence, shift,
    status="Completed"
):
    now = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO inspections
            (inspected_at, shift, image_name,
             predicted_class, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            now, shift, image_name,
            predicted_class, float(confidence), status
        ))


def get_inspections(shift="All"):
    with sqlite3.connect(DB_PATH) as conn:
        if shift == "All":
            query = """
                SELECT * FROM inspections
                ORDER BY id DESC
            """
            return pd.read_sql_query(query, conn)

        return pd.read_sql_query(
            """
            SELECT * FROM inspections
            WHERE shift = ?
            ORDER BY id DESC
            """,
            conn,
            params=(shift,)
        )
