import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

# Make sure the package root is importable when run via `streamlit run`
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facesorter.config import (config, CROP_DIR, OUTPUT_DIR, SCAN_CACHE_DB,
                               TEMP_UPLOAD_DIR)
from facesorter.face_clusterer import FaceClusterer
from facesorter.face_detector import (FaceDetector, draw_diagnostic_boxes,
                                      filter_faces)
from facesorter.file_organizer import FileOrganizer
from facesorter.media_processor import MediaProcessor
from facesorter.pipeline import group_faces, scan_files
from facesorter.scan_cache import ScanCache


# --- Cached resources (survive Streamlit reruns) ---

@st.cache_resource
def get_detector(det_size):
    """The InsightFace model load takes a few seconds; do it once."""
    return FaceDetector(det_size=det_size, crop_dir=CROP_DIR)


@st.cache_resource
def get_scan_cache():
    return ScanCache(SCAN_CACHE_DB)


# --- Helpers ---

def create_zip_archive(src_dir, zip_filepath):
    """Creates a zip archive from a source directory."""
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, src_dir))
    return zip_filepath


def reset_review_state():
    """Clears per-clustering review decisions (names, merges, deletions)."""
    st.session_state.review = {"names": {}, "merges": [], "rejected_pairs": set(),
                               "deleted": set()}


def run_scan(paths, det_size, source_desc):
    """Scans a list of files with progress UI and stores results in session."""
    if not paths:
        st.warning("No supported image files found.")
        return
    progress = st.progress(0.0, text="Scanning...")

    def on_progress(done, total, name):
        progress.progress(done / total, text=f"Scanning {done}/{total}: {name}")

    faces, stats = scan_files(
        paths,
        det_size=det_size,
        cache=get_scan_cache(),
        detector_factory=lambda: get_detector(det_size),
        progress_cb=on_progress,
    )
    progress.empty()

    st.session_state.all_faces = faces
    st.session_state.scan_stats = stats
    st.session_state.scan_source = source_desc
    st.session_state.pop("cluster_sig", None)
    reset_review_state()


def apply_merges(groups, merges):
    """Folds merged groups into their targets, following chains."""
    target_of = {}
    for src, dst in merges:
        # follow the chain in case dst was itself merged away
        while dst in target_of:
            dst = target_of[dst]
        if src != dst:
            target_of[src] = dst

    for src, dst in target_of.items():
        if src in groups and dst in groups:
            groups[dst]["faces"].extend(groups[src]["faces"])
            groups[dst]["files"] |= groups[src]["files"]
            del groups[src]
    return groups


def compute_groups(min_conf, min_height, eps, min_samples):
    """
    Filters + clusters the scanned faces. Cheap enough to run on every
    rerun, which is what makes the sliders feel instant.
    """
    faces = filter_faces(st.session_state.all_faces,
                         min_score=min_conf, min_height=min_height)
    clusterer = FaceClusterer(eps=eps, min_samples=min_samples)
    if faces:
        embeddings = np.stack([f.embedding for f in faces])
    else:
        embeddings = []
    labels, _ = clusterer.cluster_faces(embeddings)
    groups, unsorted_faces = group_faces(faces, labels)

    # Review decisions (renames, merges...) refer to cluster ids, which are
    # only stable while the clustering inputs stay the same. Reset them if
    # the settings changed.
    sig = (min_conf, min_height, eps, min_samples, len(st.session_state.all_faces))
    if st.session_state.get("cluster_sig") != sig:
        if "cluster_sig" in st.session_state and st.session_state.review != {
                "names": {}, "merges": [], "rejected_pairs": set(), "deleted": set()}:
            st.info("Clustering settings changed — group names, merges and "
                    "deletions were reset.")
        st.session_state.cluster_sig = sig
        reset_review_state()

    review = st.session_state.review
    groups = apply_merges(groups, review["merges"])
    for gid in review["deleted"]:
        groups.pop(gid, None)
    for gid, name in review["names"].items():
        if gid in groups:
            groups[gid]["name"] = name

    # Merge suggestions from centroid similarity, minus pairs already decided
    merge_candidates = []
    if len(groups) > 1:
        kept_faces = [f for g in groups.values() for f in g["faces"]]
        kept_labels = []
        for gid, g in groups.items():
            kept_labels.extend([gid] * len(g["faces"]))
        centroids = FaceClusterer.get_cluster_centroids(
            np.stack([f.embedding for f in kept_faces]), np.array(kept_labels))
        decided = review["rejected_pairs"] | {tuple(sorted(m)) for m in review["merges"]}
        merge_candidates = [
            pair for pair in FaceClusterer.find_merge_candidates(
                centroids, threshold=min(0.8, eps + 0.1))
            if tuple(sorted(pair)) not in decided
        ]

    return groups, unsorted_faces, merge_candidates


def show_face_crops(faces, columns=8, limit=None):
    """Renders a grid of face thumbnail crops."""
    shown = faces if limit is None else faces[:limit]
    cols = st.columns(columns)
    for i, face in enumerate(shown):
        if face.crop_path and os.path.exists(face.crop_path):
            cols[i % columns].image(face.crop_path, use_container_width=True)
    if limit is not None and len(faces) > limit:
        st.caption(f"...and {len(faces) - limit} more")


# --- Folder browser (server-side; the browser can't hand us local paths) ---

def _browse_to(path):
    st.session_state.browse_dir = str(path)


def _use_browse_dir():
    st.session_state.folder_input = st.session_state.browse_dir


def render_folder_browser():
    """Clickable folder navigation that fills the folder-path box."""
    browse_dir = Path(st.session_state.get("browse_dir", Path.home()))
    if not browse_dir.is_dir():
        browse_dir = Path.home()
        st.session_state.browse_dir = str(browse_dir)

    with st.container(border=True):
        up_col, home_col, path_col = st.columns([0.08, 0.08, 0.84])
        up_col.button("⬆️", help="Up one level", on_click=_browse_to,
                      args=(browse_dir.parent,),
                      disabled=browse_dir.parent == browse_dir)
        home_col.button("🏠", help="Go to your home folder",
                        on_click=_browse_to, args=(Path.home(),))
        path_col.code(str(browse_dir), language=None)

        try:
            entries = list(browse_dir.iterdir())
        except PermissionError:
            st.warning("Permission denied for this folder.")
            entries = []
        subdirs = sorted(
            (p for p in entries if p.is_dir() and not p.name.startswith('.')),
            key=lambda p: p.name.lower())
        image_count = sum(
            1 for p in entries if p.is_file()
            and p.suffix.lower() in MediaProcessor.SUPPORTED_IMAGE_FORMATS)

        st.button(f"✅ Use this folder ({image_count} images here, "
                  "subfolders included at scan)",
                  on_click=_use_browse_dir, use_container_width=True)

        max_shown = 32
        if subdirs:
            cols = st.columns(4)
            for i, sub in enumerate(subdirs[:max_shown]):
                cols[i % 4].button(
                    f"📁 {sub.name}", key=f"browse_{sub}",
                    on_click=_browse_to, args=(sub,),
                    use_container_width=True)
            if len(subdirs) > max_shown:
                st.caption(f"...and {len(subdirs) - max_shown} more subfolders "
                           "(type the path above to jump directly)")


# --- Main app sections ---

def render_input_section(det_size):
    st.subheader("1. Choose photos")
    folder_tab, upload_tab = st.tabs(["📁 Local folder", "⬆️ Upload files"])

    with folder_tab:
        if st.toggle("📂 Browse for a folder",
                     help="Navigate your folders by clicking instead of "
                          "typing a path."):
            render_folder_browser()
        folder = st.text_input(
            "Folder path",
            key="folder_input",
            placeholder="/path/to/unsorted_photos",
            help="Scanned recursively for images. Nothing is moved or "
                 "modified — sorted copies are made at export time.",
        )
        if st.button("Scan Folder", type="primary", disabled=not folder):
            try:
                paths = MediaProcessor(folder).discover_media()
            except FileNotFoundError:
                st.error(f"Folder not found: {folder}")
                return
            run_scan(paths, det_size, folder)

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose images",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Scan Uploads", type="primary"):
            upload_dir = Path(TEMP_UPLOAD_DIR)
            shutil.rmtree(upload_dir, ignore_errors=True)
            upload_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for file in uploaded:
                dest = upload_dir / os.path.basename(file.name)
                with open(dest, "wb") as f:
                    shutil.copyfileobj(file, f)
                paths.append(dest)
            run_scan(paths, det_size, f"{len(paths)} uploaded files")


def render_groups_section(groups, unsorted_faces, merge_candidates):
    st.subheader("2. Review groups")
    stats = st.session_state.scan_stats
    total_faces = sum(len(g["faces"]) for g in groups.values())
    st.caption(
        f"Source: {st.session_state.scan_source} — "
        f"{stats['total']} files ({stats['cached']} from cache, "
        f"{stats['unreadable']} unreadable) · "
        f"{total_faces} faces in {len(groups)} groups, "
        f"{len(unsorted_faces)} unsorted."
    )

    # --- Merge suggestions ---
    if merge_candidates:
        with st.container(border=True):
            st.markdown("**Suggested merges** — these groups look like the "
                        "same person:")
            with st.form("merge_form"):
                for i, (id1, id2) in enumerate(merge_candidates):
                    col1, col2, col3 = st.columns([0.15, 0.15, 0.7])
                    for col, gid in ((col1, id1), (col2, id2)):
                        rep = groups[gid]["rep_face"]
                        if rep.crop_path and os.path.exists(rep.crop_path):
                            col.image(rep.crop_path, use_container_width=True)
                        col.caption(groups[gid]["name"])
                    col3.radio(
                        f"Merge **{groups[id2]['name']}** into "
                        f"**{groups[id1]['name']}**?",
                        options=["Skip", "Merge", "Not the same person"],
                        key=f"merge_choice_{id1}_{id2}",
                        horizontal=True,
                    )
                if st.form_submit_button("Apply merge decisions"):
                    review = st.session_state.review
                    for id1, id2 in merge_candidates:
                        choice = st.session_state.get(f"merge_choice_{id1}_{id2}")
                        if choice == "Merge":
                            review["merges"].append((id2, id1))
                        elif choice == "Not the same person":
                            review["rejected_pairs"].add(tuple(sorted((id1, id2))))
                    st.rerun()

    # --- Group list with rename/delete ---
    sig_key = hashlib.sha1(str(st.session_state.cluster_sig).encode()).hexdigest()[:8]
    with st.form("groups_form"):
        st.caption("Rename groups or mark them for removal, then apply. "
                   "Giving two groups the same name combines them at export.")
        for gid, group in groups.items():
            with st.container(border=True):
                col1, col2 = st.columns([0.12, 0.88])
                with col1:
                    rep = group["rep_face"]
                    if rep.crop_path and os.path.exists(rep.crop_path):
                        st.image(rep.crop_path, use_container_width=True)
                with col2:
                    name_col, del_col = st.columns([0.72, 0.28])
                    name_col.text_input(
                        "Group name", value=group["name"],
                        key=f"name_{sig_key}_{gid}",
                        label_visibility="collapsed",
                    )
                    del_col.checkbox("Remove", key=f"delete_{sig_key}_{gid}",
                                     help="Don't export this group.")
                    st.caption(f"{len(group['faces'])} faces in "
                               f"{len(group['files'])} photos")
                    with st.expander("Show faces and files"):
                        show_face_crops(group["faces"], limit=24)
                        st.code("\n".join(sorted(
                            os.path.basename(p) for p in group["files"])))

        if st.form_submit_button("Apply changes", type="primary",
                                 use_container_width=True):
            review = st.session_state.review
            for gid, group in groups.items():
                if st.session_state.get(f"delete_{sig_key}_{gid}"):
                    review["deleted"].add(gid)
                    continue
                raw = st.session_state.get(f"name_{sig_key}_{gid}", "")
                sanitized = "".join(
                    c for c in raw if c.isalnum() or c in (' ', '_', '-')).strip()
                if sanitized and sanitized != group["name"]:
                    review["names"][gid] = sanitized
            st.rerun()

    # --- Unsorted faces ---
    if unsorted_faces:
        with st.expander(f"🫥 Unsorted faces ({len(unsorted_faces)}) — didn't "
                         "match any group"):
            st.caption("Usually tiny/blurry faces or one-off detections. "
                       "Raise 'Cluster distance' or lower 'Min faces per "
                       "group' to pull more of these into groups.")
            show_face_crops(unsorted_faces, limit=48)


def render_export_section(groups):
    st.subheader("3. Export")
    output_dir = st.text_input("Output folder", value=OUTPUT_DIR)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export sorted folders", type="primary",
                     use_container_width=True, disabled=not groups):
            # Groups sharing a name are intentionally combined into one folder
            merged_by_name = {}
            for gid, group in groups.items():
                entry = merged_by_name.setdefault(
                    group["name"], {"name": group["name"], "files": set()})
                entry["files"] |= group["files"]
            with st.spinner("Copying files..."):
                copied = FileOrganizer(output_dir).export(
                    {name: g for name, g in merged_by_name.items()})
            total = sum(copied.values())
            st.session_state.exported_dir = output_dir
            st.success(f"Copied {total} files into {len(copied)} folders "
                       f"under {output_dir}")

    with col2:
        exported = st.session_state.get("exported_dir")
        if exported and os.path.isdir(exported):
            zip_path = os.path.join(os.path.dirname(exported) or ".",
                                    "sorted_photos.zip")
            if st.button("Create ZIP of export", use_container_width=True):
                with st.spinner("Zipping..."):
                    create_zip_archive(exported, zip_path)
                st.session_state.zip_ready = zip_path
            zip_ready = st.session_state.get("zip_ready")
            if zip_ready and os.path.exists(zip_ready):
                with open(zip_ready, "rb") as fp:
                    st.download_button("Download ZIP", data=fp,
                                       file_name="sorted_photos.zip",
                                       mime="application/zip",
                                       use_container_width=True)


def run_diagnostic_tool(det_size, min_conf, min_height):
    st.header("Face Detection Diagnostic Tool")
    st.info("Upload one photo to see exactly what the detector finds. "
            "Green boxes pass your current sidebar filters; red boxes were "
            "detected but filtered out. Use this to tune confidence and "
            "face-size settings for your photo conditions.")

    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded is None:
        return

    temp_dir = Path(TEMP_UPLOAD_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"diagnostic_{os.path.basename(uploaded.name)}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Detecting faces..."):
        detector = get_detector(det_size)
        faces = detector.detect_faces(temp_path)

    if faces is None:
        st.error("Could not read the uploaded image. It might be corrupt.")
        return
    if not faces:
        st.warning("No faces detected at all. Try a higher detector "
                   "resolution in the sidebar.")
        return

    bgr = FaceDetector.load_image_bgr(temp_path)
    st.image(draw_diagnostic_boxes(bgr, faces, min_conf, min_height),
             use_container_width=True)

    passing = filter_faces(faces, min_conf, min_height)
    st.metric("Faces passing filters", f"{len(passing)} / {len(faces)}")
    for i, face in enumerate(faces):
        ok = face.score >= min_conf and face.height >= min_height
        status = "✅ kept" if ok else "❌ filtered out"
        st.markdown(f"**Face {i + 1}**: confidence `{face.score:.2f}`, "
                    f"height `{int(face.height)}px` ({status})")


def main():
    st.set_page_config(page_title="FaceSorter", layout="wide")

    if "review" not in st.session_state:
        reset_review_state()

    # --- Sidebar ---
    st.sidebar.title("⚙️ Settings")
    app_mode = st.sidebar.radio("App mode", ("Face Sorter", "Diagnostic Tool"))

    st.sidebar.subheader("Detection (applies at scan time)")
    det_size = st.sidebar.select_slider(
        "Detector resolution", options=[640, 1024, 1600],
        value=config.get("detection.det_size", 640),
        help="Higher finds smaller/further faces but scans slower. "
             "Changing this requires a re-scan.")

    st.sidebar.subheader("Face filters (instant)")
    min_conf = st.sidebar.slider(
        "Min detection confidence", 0.30, 0.90,
        value=float(config.get("detection.min_confidence", 0.5)), step=0.01,
        help="Faces the detector is less sure about are ignored. Lower this "
             "if real faces are being missed; raise it if non-faces slip in.")
    min_height = st.sidebar.slider(
        "Min face height (px)", 0, 500,
        value=int(config.get("detection.min_face_height", 40)), step=5,
        help="Ignore faces smaller than this — background strangers, "
             "photo-bombers, faces on posters.")

    st.sidebar.subheader("Grouping (instant)")
    eps = st.sidebar.slider(
        "Cluster distance (eps)", 0.20, 0.80,
        value=float(config.get("clustering.eps", 0.5)), step=0.01,
        help="How similar two faces must be to be the same person. "
             "If one person is split across groups, increase it. "
             "If different people share a group, decrease it.")
    min_samples = st.sidebar.slider(
        "Min faces per group", 1, 5,
        value=int(config.get("clustering.min_samples", 2)),
        help="Groups need at least this many faces; loners go to 'Unsorted'. "
             "Set to 1 to give every face a group.")

    if st.sidebar.button("Clear scan cache",
                         help="Forget all cached scans; next scan re-detects "
                              "everything from scratch."):
        get_scan_cache().clear()
        st.sidebar.success("Scan cache cleared.")

    if app_mode == "Diagnostic Tool":
        run_diagnostic_tool(det_size, min_conf, min_height)
        return

    # --- Face Sorter flow ---
    st.title("📷 FaceSorter")
    st.write("Point FaceSorter at your photos and it sorts them into a folder "
             "per person — group photos are copied into every member's folder.")

    render_input_section(det_size)

    if st.session_state.get("all_faces") is None:
        return
    if not st.session_state.all_faces:
        st.warning("No faces were found in the scanned photos.")
        return

    groups, unsorted_faces, merge_candidates = compute_groups(
        min_conf, min_height, eps, min_samples)

    st.write("---")
    render_groups_section(groups, unsorted_faces, merge_candidates)
    st.write("---")
    render_export_section(groups)


if __name__ == "__main__":
    main()
