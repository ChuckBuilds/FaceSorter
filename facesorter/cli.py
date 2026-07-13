import argparse
import sys

import numpy as np

from facesorter.config import config, CROP_DIR, SCAN_CACHE_DB
from facesorter.face_clusterer import FaceClusterer
from facesorter.face_detector import FaceDetector, filter_faces
from facesorter.file_organizer import FileOrganizer
from facesorter.media_processor import MediaProcessor
from facesorter.pipeline import group_faces, scan_files
from facesorter.scan_cache import ScanCache


def main():
    """Headless run of the full scan -> cluster -> export pipeline."""
    parser = argparse.ArgumentParser(
        description="Sort a folder of photos into per-person folders based "
                    "on the faces in them.")
    parser.add_argument("--source", required=True,
                        help="Folder of photos to sort (scanned recursively).")
    parser.add_argument("--output", default=config.get("output.dir", "sorted_output"),
                        help="Folder to copy the sorted photos into.")
    parser.add_argument("--det-size", type=int,
                        default=config.get("detection.det_size", 640),
                        choices=[640, 1024, 1600],
                        help="Detector resolution; higher finds smaller faces.")
    parser.add_argument("--min-confidence", type=float,
                        default=config.get("detection.min_confidence", 0.5),
                        help="Ignore faces below this detector confidence.")
    parser.add_argument("--min-face-height", type=int,
                        default=config.get("detection.min_face_height", 40),
                        help="Ignore faces shorter than this many pixels.")
    parser.add_argument("--eps", type=float,
                        default=config.get("clustering.eps", 0.5),
                        help="Max cosine distance to group two faces "
                             "(lower = stricter).")
    parser.add_argument("--min-samples", type=int,
                        default=config.get("clustering.min_samples", 2),
                        help="Minimum faces to form a group.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore the scan cache and re-detect everything.")
    args = parser.parse_args()

    try:
        paths = MediaProcessor(args.source).discover_media()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not paths:
        sys.exit(f"No supported images found in {args.source}")
    print(f"Found {len(paths)} images in {args.source}")

    cache = None if args.no_cache else ScanCache(SCAN_CACHE_DB)

    def on_progress(done, total, name):
        print(f"\r  Scanning {done}/{total}: {name[:50]:<50}", end="", flush=True)

    faces, stats = scan_files(
        paths,
        det_size=args.det_size,
        cache=cache,
        detector_factory=lambda: FaceDetector(det_size=args.det_size,
                                              crop_dir=CROP_DIR),
        progress_cb=on_progress,
    )
    print()
    print(f"  {len(faces)} faces found "
          f"({stats['cached']} files from cache, "
          f"{stats['unreadable']} unreadable)")

    faces = filter_faces(faces, min_score=args.min_confidence,
                         min_height=args.min_face_height)
    if not faces:
        sys.exit("No faces passed the filters — try lowering "
                 "--min-confidence or --min-face-height.")

    embeddings = np.stack([f.embedding for f in faces])
    labels, num_clusters = FaceClusterer(
        eps=args.eps, min_samples=args.min_samples).cluster_faces(embeddings)
    groups, unsorted_faces = group_faces(faces, labels)
    print(f"  {len(faces)} faces -> {num_clusters} people "
          f"({len(unsorted_faces)} unsorted faces)")

    copied = FileOrganizer(args.output).export(groups)
    for gid, group in groups.items():
        print(f"  {group['name']}: {len(group['files'])} photos "
              f"({copied[gid]} copied)")
    print(f"Done. Sorted folders are in {args.output}")


if __name__ == "__main__":
    main()
