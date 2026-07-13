"""Shared scan/cluster plumbing used by both the Streamlit app and the CLI."""
from collections import defaultdict
from pathlib import Path

from facesorter.face_clusterer import UNSORTED_LABEL


def scan_files(paths, det_size, cache=None, detector_factory=None,
               progress_cb=None):
    """
    Detects and embeds faces in every file, using the cache where possible.

    The detector (a slow model load) is only created on the first cache
    miss, so a fully-cached re-scan is nearly instant.

    Args:
        paths: iterable of image paths.
        det_size (int): detector input resolution (part of the cache key).
        cache (ScanCache, optional): disk cache of previous scans.
        detector_factory (callable): returns a FaceDetector; called lazily.
        progress_cb (callable, optional): progress_cb(done, total, filename).

    Returns:
        (faces, stats): flat list of DetectedFace across all files, and a
        dict with 'cached' / 'scanned' / 'unreadable' counts.
    """
    paths = list(paths)
    faces = []
    stats = {"total": len(paths), "cached": 0, "scanned": 0, "unreadable": 0}
    detector = None

    for i, path in enumerate(paths):
        cached = cache.get(path, det_size) if cache else None
        if cached is not None:
            faces.extend(cached)
            stats["cached"] += 1
        else:
            if detector is None:
                detector = detector_factory()
            result = detector.detect_faces(path)
            if result is None:
                stats["unreadable"] += 1
            else:
                faces.extend(result)
                stats["scanned"] += 1
                if cache:
                    cache.put(path, det_size, result)
        if progress_cb:
            progress_cb(i + 1, len(paths), Path(path).name)

    return faces, stats


def group_faces(faces, labels):
    """
    Organizes clustered faces into per-person groups.

    Returns:
        (groups, unsorted): groups is {label: {"name", "faces", "files",
        "rep_face"}} sorted so bigger groups come first; unsorted is the
        list of noise faces (label -1).
    """
    by_label = defaultdict(list)
    for face, label in zip(faces, labels):
        by_label[int(label)].append(face)

    unsorted = by_label.pop(UNSORTED_LABEL, [])

    groups = {}
    order = sorted(by_label, key=lambda lb: -len(by_label[lb]))
    for label in order:
        members = by_label[label]
        groups[label] = {
            "name": f"Person_{label + 1}",
            "faces": members,
            "files": {f.source_path for f in members},
            "rep_face": max(members, key=lambda f: f.score),
        }
    return groups, unsorted
