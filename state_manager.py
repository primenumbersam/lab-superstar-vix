import sqlite3
import json
import os

class StateManager:
    """Manages EWMA and index state using SQLite."""
    def __init__(self, db_path="state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            conn.commit()

    def set(self, key, value):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('REPLACE INTO state (key, value) VALUES (?, ?)', (key, json.dumps(value)))
            conn.commit()

    def get(self, key, default=None):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT value FROM state WHERE key = ?', (key,))
            row = c.fetchone()
            if row:
                return json.loads(row[0])
            return default
