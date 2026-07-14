"""Shared scan/cluster plumbing used by both the Streamlit app and the CLI."""
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def match_known_people(faces, people_centroids, threshold):
    """
    Matches faces against saved people before clustering.

    Args:
        faces: list of DetectedFace.
        people_centroids: {person_id: (name, normalized centroid)} from
            PeopleDB.centroids().
        threshold (float): max cosine distance to a person's centroid.

    Returns:
        (matched, remaining): matched is {person_id: [DetectedFace, ...]};
        remaining is the list of faces left for DBSCAN.
    """
    if not people_centroids or not faces:
        return {}, list(faces)

    ids = list(people_centroids)
    centroid_matrix = np.stack([people_centroids[pid][1] for pid in ids])
    embeddings = np.stack([f.embedding for f in faces])
    # cosine distance between unit vectors, all faces x all people at once
    distances = 1.0 - embeddings @ centroid_matrix.T

    matched = defaultdict(list)
    remaining = []
    for i, face in enumerate(faces):
        best = int(np.argmin(distances[i]))
        if distances[i][best] <= threshold:
            matched[ids[best]].append(face)
        else:
            remaining.append(face)
    return dict(matched), remaining


def make_group(name, members):
    embeddings = np.stack([f.embedding for f in members])
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    centroid = mean / norm if norm > 0 else mean
    # order members by similarity to the group centroid so the most
    # questionable matches sit at the end of the strip
    dists = 1.0 - embeddings @ centroid
    order = np.argsort(dists)
    return {
        "name": name,
        "faces": [members[i] for i in order],
        "face_dists": [float(dists[i]) for i in order],
        "files": {f.source_path for f in members},
        "rep_face": members[int(order[0])],
        "centroid": centroid,
        "known": False,
    }


def group_faces(faces, labels, known_matches=None, people_names=None):
    """
    Organizes faces into per-person groups.

    Group ids are strings: "p<person_id>" for saved-people matches,
    "c<label>" for fresh clusters. Within each group, faces are sorted by
    distance to the group centroid (best match first).

    Args:
        faces / labels: clustering input and DBSCAN output, aligned.
        known_matches: {person_id: [DetectedFace]} from match_known_people.
        people_names: {person_id: name} for naming known groups.

    Returns:
        (groups, unsorted): groups is {group_id: {"name", "faces",
        "face_dists", "files", "rep_face", "centroid", "known"}} with known
        people first, then bigger clusters first; unsorted is the list of
        noise faces (label -1).
    """
    groups = {}
    for pid, members in sorted((known_matches or {}).items(),
                               key=lambda kv: -len(kv[1])):
        group = make_group(people_names[pid], members)
        group["known"] = True
        groups[f"p{pid}"] = group

    by_label = defaultdict(list)
    for face, label in zip(faces, labels):
        by_label[int(label)].append(face)
    unsorted = by_label.pop(UNSORTED_LABEL, [])

    for label in sorted(by_label, key=lambda lb: -len(by_label[lb])):
        groups[f"c{label}"] = make_group(f"Person_{label + 1}",
                                          by_label[label])
    return groups, unsorted


def suggest_rescues(groups, unsorted, threshold):
    """
    Finds unsorted faces close enough to an existing group to be worth
    suggesting ("might also be George"). Each face is offered to its
    nearest group only.

    Returns:
        {group_id: [(DetectedFace, distance), ...]} sorted best-first.
    """
    if not groups or not unsorted:
        return {}
    ids = list(groups)
    centroid_matrix = np.stack([groups[g]["centroid"] for g in ids])
    embeddings = np.stack([f.embedding for f in unsorted])
    distances = 1.0 - embeddings @ centroid_matrix.T

    rescues = defaultdict(list)
    for i, face in enumerate(unsorted):
        best = int(np.argmin(distances[i]))
        if distances[i][best] <= threshold:
            rescues[ids[best]].append((face, float(distances[i][best])))
    return {gid: sorted(lst, key=lambda t: t[1]) for gid, lst in rescues.items()}
