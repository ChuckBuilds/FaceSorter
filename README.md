# FaceSorter

FaceSorter sorts a folder of unsorted photos into a folder per person, based on the faces in each photo. A group photo of Alice, Bob and Carol is copied into all three of their folders, so every person's folder contains every photo they appear in — ready to share.

Face detection and recognition run locally via [InsightFace](https://github.com/deepinsight/insightface) (SCRFD detector + ArcFace embeddings) on CPU — no GPU, CUDA, or compiler toolchain required.

## Quick start

Requires Python 3.10+ (tested on 3.12 and 3.14). No compiler, CMake, or GPU needed.

```bash
git clone https://github.com/ChuckBuilds/FaceSorter
cd FaceSorter
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.in
streamlit run facesorter/app.py
```

The app opens at <http://localhost:8501>. On the very first scan, InsightFace downloads its models (~300 MB, one time) to `~/.insightface` — expect a short delay before the progress bar starts moving.

## Usage

### Web app

```bash
source venv/bin/activate        # if not already active
streamlit run facesorter/app.py
```

1. **Choose photos** — point it at a local folder (scanned recursively; nothing in it is modified) by typing a path or using the built-in **📂 Browse for a folder** toggle to click through your folders, or drag-and-drop uploads.
2. **Scan** — every face is detected and fingerprinted once, then cached on disk (`~/.facesorter/scan_cache.db`). Re-scanning the same folder is nearly instant and only processes new or changed files.
3. **Review groups** — faces are grouped into people. Rename groups, apply suggested merges, remove groups you don't want, or untick individual faces that don't belong. Inside each group, faces are sorted best-match-first, so a glance at the end of the strip catches mistakes. Unsorted faces that sit close to a group are offered back as one-click "might also be this person" suggestions. All the tuning sliders re-group instantly without re-scanning.
4. **Save people** — tick **💾 Save person** on a group to remember them permanently. Future scans recognize saved people automatically and name their folders — labeling effort accumulates instead of resetting. Manage saved people in the sidebar.
5. **Export** — copies of the originals are written to `<output>/<person>/`, with an optional ZIP.

### Command line

```bash
source venv/bin/activate        # if not already active
python -m facesorter.cli --source /path/to/photos --output sorted_output
```

Run `python -m facesorter.cli --help` for all tuning flags (`--eps`, `--min-confidence`, etc. — same meanings as the app's sliders).

### Docker

```bash
docker build -t facesorter .
docker run -p 8501:8501 \
  -v ~/Pictures:/photos:ro \
  -v ~/facesorter-output:/output \
  -v facesorter-models:/root/.insightface \
  -v facesorter-data:/root/.facesorter \
  facesorter
```

Then open <http://localhost:8501>, use `/photos` (or any subfolder) as the folder path, and set the output folder to `/output` when exporting. The named volumes persist the downloaded models and the scan cache across container restarts.

## Understanding the settings

**Detector resolution** *(re-scan required)* — the size photos are analyzed at. 640 is fast and fine for typical photos; 1024/1600 find smaller and more distant faces at the cost of scan speed.

**Min detection confidence** — how sure the detector must be that something is a face. Lower it if real faces are missed; raise it if non-faces (statues, posters, pareidolia) sneak in.

**Min face height (px)** — ignores small background faces: strangers, crowds, photo-bombers. Use the **Diagnostic Tool** (sidebar) on a sample photo to see each detected face's size and confidence and pick sensible values for your photos.

**Cluster distance (eps)** — the most important dial. How similar two face fingerprints must be to count as the same person:
- One person split across multiple groups → **increase** it (e.g. 0.50 → 0.55)
- Different people lumped into one group → **decrease** it (e.g. 0.50 → 0.45)

Because scans are cached, moving this slider re-groups instantly — tune it freely.

**Min faces per group** — groups smaller than this go to the *Unsorted* bucket instead of becoming their own folder. The default of 2 keeps one-off false detections from creating junk folders; set it to 1 if you want singletons too.

**Saved-person match distance** — how close a face must be to a saved person's average faceprint to be auto-recognized. Slightly stricter than `eps` by default (0.45). Raise it if a saved person's new photos aren't being recognized; lower it if the wrong photos are landing in their folder.

## Notes

- Exporting **copies** files; your originals are never moved or altered.
- Giving two groups the same name combines them into one folder at export.
- HEIC/HEIF photos (iPhone) are supported via `pillow-heif`.
- App data (scan cache, face thumbnails, saved people) lives in `~/.facesorter/` — the scan cache is safe to delete anytime (*Clear scan cache* button); deleting `people.db` forgets your saved people.
- The CLI also recognizes saved people (`--ignore-people` to opt out), so a fully hands-off re-sort is: scan once in the app, label people, then just run the CLI whenever new photos arrive.
