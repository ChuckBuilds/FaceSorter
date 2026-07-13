import io
import os
import sqlite3
from pathlib import Path

import numpy as np

from facesorter.face_detector import DetectedFace


class ScanCache:
    """
    Disk cache of face-scan results, keyed by (path, detector size) and
    validated against the file's mtime. Re-scanning a folder only processes
    new or modified files, and tuning sliders never triggers a re-scan.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                path TEXT NOT NULL,
                det_size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                faces BLOB NOT NULL,
                PRIMARY KEY (path, det_size)
            )
            """
        )
        self._conn.commit()

    def get(self, image_path, det_size):
        """
        Returns the cached list[DetectedFace] for the file, or None on a
        cache miss (never scanned, or file modified since).
        """
        path = str(image_path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        row = self._conn.execute(
            "SELECT mtime, faces FROM scans WHERE path = ? AND det_size = ?",
            (path, int(det_size)),
        ).fetchone()
        if row is None or row[0] != mtime:
            return None
        return _deserialize_faces(path, row[1])

    def put(self, image_path, det_size, faces):
        """Stores scan results for a file."""
        path = str(image_path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO scans (path, det_size, mtime, faces) "
            "VALUES (?, ?, ?, ?)",
            (path, int(det_size), mtime, _serialize_faces(faces)),
        )
        self._conn.commit()

    def clear(self):
        """Empties the cache entirely."""
        self._conn.execute("DELETE FROM scans")
        self._conn.commit()

    def close(self):
        self._conn.close()


def _serialize_faces(faces):
    buf = io.BytesIO()
    n = len(faces)
    np.savez_compressed(
        buf,
        bboxes=np.stack([f.bbox for f in faces]) if n else np.zeros((0, 4), np.float32),
        scores=np.array([f.score for f in faces], np.float32),
        embeddings=np.stack([f.embedding for f in faces]) if n else np.zeros((0, 512), np.float32),
        crop_paths=np.array([f.crop_path for f in faces], dtype=object) if n else np.array([], dtype=object),
    )
    return buf.getvalue()


def _deserialize_faces(source_path, blob):
    with np.load(io.BytesIO(blob), allow_pickle=True) as data:
        return [
            DetectedFace(
                source_path=source_path,
                bbox=data["bboxes"][i],
                score=float(data["scores"][i]),
                embedding=data["embeddings"][i],
                crop_path=str(data["crop_paths"][i]),
            )
            for i in range(len(data["scores"]))
        ]
