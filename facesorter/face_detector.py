import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Detection threshold used at scan time. Kept intentionally low so the cache
# stores every plausible face; the UI's confidence slider filters after the
# fact without invalidating cached scans.
SCAN_DET_THRESH = 0.30


@dataclass
class DetectedFace:
    """A single face found in an image."""
    source_path: str          # original media file the face came from
    bbox: np.ndarray          # (4,) float32: x1, y1, x2, y2 in original image coords
    score: float              # detector confidence, 0-1
    embedding: np.ndarray     # (512,) float32, L2-normalized ArcFace embedding
    crop_path: str = field(default="")  # thumbnail of the face, for the UI

    @property
    def height(self):
        return float(self.bbox[3] - self.bbox[1])


class FaceDetector:
    """
    Face detection + embedding via InsightFace (SCRFD detector, ArcFace
    recognition). Models are downloaded to ~/.insightface on first use.
    """

    def __init__(self, det_size=640, crop_dir=None):
        """
        Args:
            det_size (int): Detector input resolution. Higher finds smaller
                            faces but is slower (640 is a good default).
            crop_dir (str or Path, optional): Directory to save face thumbnail
                            crops into. Crops are skipped if omitted.
        """
        from insightface.app import FaceAnalysis  # deferred: slow import

        self.det_size = int(det_size)
        self.crop_dir = Path(crop_dir) if crop_dir else None
        if self.crop_dir:
            self.crop_dir.mkdir(parents=True, exist_ok=True)

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size),
                         det_thresh=SCAN_DET_THRESH)

    @staticmethod
    def load_image_bgr(image_path):
        """
        Loads an image as a BGR numpy array, honoring EXIF orientation
        (phone photos are often stored rotated). Returns None on failure.
        """
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                rgb = np.asarray(img.convert("RGB"))
            return rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV/InsightFace
        except Exception:
            return None

    def detect_faces(self, image_path):
        """
        Detects and embeds every face in an image.

        Returns:
            list[DetectedFace], or None if the image could not be read.
        """
        bgr = self.load_image_bgr(image_path)
        if bgr is None:
            return None
        faces = self.app.get(bgr)

        results = []
        for face in faces:
            detected = DetectedFace(
                source_path=str(image_path),
                bbox=face.bbox.astype(np.float32),
                score=float(face.det_score),
                embedding=face.normed_embedding.astype(np.float32),
            )
            if self.crop_dir is not None:
                detected.crop_path = self._save_crop(bgr, detected)
            results.append(detected)
        return results

    def _save_crop(self, bgr, face, padding_frac=0.25, thumb_size=160):
        """Saves a padded thumbnail crop of the face; returns its path."""
        crop_path = self.crop_dir / f"{crop_key(face)}.jpg"
        if crop_path.exists():
            return str(crop_path)

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = face.bbox
        pad = padding_frac * max(x2 - x1, y2 - y1)
        x1 = max(0, int(x1 - pad))
        y1 = max(0, int(y1 - pad))
        x2 = min(w, int(x2 + pad))
        y2 = min(h, int(y2 + pad))
        if x2 <= x1 or y2 <= y1:
            return ""

        crop = bgr[y1:y2, x1:x2]
        scale = thumb_size / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                     max(1, int(crop.shape[0] * scale))))
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(crop_path)


def crop_key(face):
    """Stable id for a face's thumbnail, derived from its source and bbox."""
    return hashlib.sha1(
        f"{face.source_path}|{face.bbox.round(1).tolist()}".encode()
    ).hexdigest()


def filter_faces(faces, min_score=0.5, min_height=40):
    """
    Applies the UI's post-detection filters to a list of DetectedFace.
    Kept separate from detection so changing filters never re-scans.
    """
    return [f for f in faces if f.score >= min_score and f.height >= min_height]


def draw_diagnostic_boxes(bgr, faces, min_score, min_height):
    """
    Draws labeled boxes on a copy of the image for the diagnostic tool.
    Green = passes the current filters, red = detected but filtered out.
    Returns an RGB numpy array for display.
    """
    canvas = bgr.copy()
    thickness = max(2, canvas.shape[1] // 640)
    font_scale = max(0.5, canvas.shape[1] / 1600)
    for i, face in enumerate(faces):
        ok = face.score >= min_score and face.height >= min_height
        color = (0, 200, 0) if ok else (0, 0, 230)
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"#{i + 1} conf {face.score:.2f} h {int(face.height)}px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, thickness)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(canvas, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness)
    return canvas[:, :, ::-1]
