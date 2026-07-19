import sqlite3
import time
from pathlib import Path

import numpy as np

from facesorter.face_detector import crop_key

EMBEDDING_DIM = 512


class PeopleDB:
    """
    Persistent database of named people and their enrolled faceprints.

    Once a group is saved under a name, future scans match new faces
    against each person's centroid and auto-name their folder, so labeling
    effort accumulates across sessions instead of resetting.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faceprints (
                face_key TEXT PRIMARY KEY,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                crop_path TEXT,
                added REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_faceprints_person
                ON faceprints(person_id);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO meta VALUES ('version', 0);
            """
        )
        self._conn.commit()

    def _bump(self):
        self._conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'version'")
        self._conn.commit()

    def version(self):
        """Monotonic counter, bumped on every change. Cheap staleness check."""
        return self._conn.execute(
            "SELECT value FROM meta WHERE key = 'version'").fetchone()[0]

    def add_or_get_person(self, name):
        """Returns the id for `name`, creating the person if needed."""
        row = self._conn.execute(
            "SELECT id FROM people WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        cur = self._conn.execute(
            "INSERT INTO people (name, created) VALUES (?, ?)",
            (name, time.time()))
        self._bump()
        return cur.lastrowid

    def rename_person(self, person_id, new_name):
        """Renames a person. Raises sqlite3.IntegrityError if name is taken."""
        self._conn.execute(
            "UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))
        self._bump()

    def delete_person(self, person_id):
        self._conn.execute("DELETE FROM faceprints WHERE person_id = ?",
                           (person_id,))
        self._conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        self._bump()

    def enroll_faces(self, person_id, faces):
        """
        Adds faceprints for a person. Faces already enrolled (same photo,
        same box) are skipped, so re-saving a group is idempotent.
        """
        now = time.time()
        self._conn.executemany(
            "INSERT OR IGNORE INTO faceprints "
            "(face_key, person_id, embedding, crop_path, added) "
            "VALUES (?, ?, ?, ?, ?)",
            [(crop_key(f), person_id,
              f.embedding.astype(np.float32).tobytes(), f.crop_path, now)
             for f in faces])
        self._bump()

    def list_people(self):
        """Returns [{id, name, n_faces, sample_crop}] sorted by name."""
        rows = self._conn.execute(
            """
            SELECT p.id, p.name, COUNT(f.face_key),
                   MAX(CASE WHEN f.crop_path != '' THEN f.crop_path END)
            FROM people p LEFT JOIN faceprints f ON f.person_id = p.id
            GROUP BY p.id ORDER BY p.name COLLATE NOCASE
            """).fetchall()
        return [{"id": r[0], "name": r[1], "n_faces": r[2], "sample_crop": r[3]}
                for r in rows]

    def centroids(self):
        """
        Returns {person_id: (name, normalized centroid)} for people with at
        least one faceprint.
        """
        result = {}
        for pid, name in self._conn.execute("SELECT id, name FROM people"):
            blobs = self._conn.execute(
                "SELECT embedding FROM faceprints WHERE person_id = ?",
                (pid,)).fetchall()
            if not blobs:
                continue
            embeddings = np.stack([
                np.frombuffer(b[0], dtype=np.float32) for b in blobs])
            mean = embeddings.mean(axis=0)
            norm = np.linalg.norm(mean)
            result[pid] = (name, mean / norm if norm > 0 else mean)
        return result

    def close(self):
        self._conn.close()
