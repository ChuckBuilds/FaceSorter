import face_recognition
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

class FaceQuality(Enum):
    EXCELLENT = 4
    GOOD = 3
    FAIR = 2
    POOR = 1

@dataclass
class FaceInfo:
    encoding: np.ndarray
    location: Tuple[int, int, int, int]  # top, right, bottom, left
    quality_score: float
    pose_score: float  # How frontal the face is (0-1, 1 being perfectly frontal)
    sharpness_score: float
    lighting_score: float
    area: int
    confidence: float
    quality_level: FaceQuality

class ImprovedFaceDetector:
    """
    Advanced face detection with quality assessment, pose estimation, and adaptive preprocessing.
    """

    def __init__(self, model="hog", enable_gpu=False):
        """
        Initialize the improved face detector.
        
        Args:
            model: Detection model ("hog", "cnn", or "adaptive")
            enable_gpu: Whether to use GPU acceleration if available
        """
        self.model = model
        self.enable_gpu = enable_gpu
        self.face_cascade = None
        
        # Initialize OpenCV cascade for additional validation
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            logging.warning("Could not load OpenCV face cascade")
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_sharpness': 50.0,
            'min_lighting': 30.0,
            'min_pose_score': 0.3,
            'min_area': 2000,
            'min_confidence': 0.5
        }

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive image enhancement for better face detection.
        """
        pil_image = Image.fromarray(image)
        
        # Auto-contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Adaptive brightness adjustment
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Slight sharpening
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return np.array(pil_image)

    def calculate_face_quality(self, image: np.ndarray, location: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for a detected face.
        """
        top, right, bottom, left = location
        face_region = image[top:bottom, left:right]
        
        if face_region.size == 0:
            return {
                'sharpness': 0.0,
                'lighting': 0.0,
                'pose_score': 0.0,
                'overall_quality': 0.0
            }
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        
        # 1. Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # 2. Lighting quality (avoid over/under exposure)
        mean_brightness = np.mean(gray_face)
        brightness_variance = np.var(gray_face)
        lighting_score = min(100, brightness_variance) * (1 - abs(mean_brightness - 127) / 127)
        
        # 3. Pose estimation using facial landmarks
        pose_score = self._estimate_pose_quality(image, location)
        
        # 4. Overall quality score
        normalized_sharpness = min(100, sharpness) / 100
        normalized_lighting = lighting_score / 100
        overall_quality = (normalized_sharpness * 0.4 + normalized_lighting * 0.3 + pose_score * 0.3)
        
        return {
            'sharpness': sharpness,
            'lighting': lighting_score,
            'pose_score': pose_score,
            'overall_quality': overall_quality
        }

    def _estimate_pose_quality(self, image: np.ndarray, location: Tuple[int, int, int, int]) -> float:
        """
        Estimate how frontal/good the face pose is using facial landmarks.
        """
        try:
            # Get facial landmarks
            landmarks = face_recognition.face_landmarks(image, [location])
            if not landmarks:
                return 0.5  # Default moderate score
            
            landmark_points = landmarks[0]
            
            # Calculate symmetry of key facial features
            if 'left_eye' in landmark_points and 'right_eye' in landmark_points:
                left_eye_center = np.mean(landmark_points['left_eye'], axis=0)
                right_eye_center = np.mean(landmark_points['right_eye'], axis=0)
                
                # Eye level difference (should be minimal for frontal faces)
                eye_level_diff = abs(left_eye_center[1] - right_eye_center[1])
                eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
                
                if eye_distance > 0:
                    symmetry_score = 1.0 - min(1.0, eye_level_diff / (eye_distance * 0.1))
                    return max(0.0, symmetry_score)
            
            return 0.5
        except:
            return 0.5

    def detect_faces_adaptive(self, image_path: str, min_face_area: Optional[int] = None) -> Tuple[np.ndarray, List[FaceInfo], List[Tuple]]:
        """
        Advanced face detection with quality assessment and adaptive processing.
        """
        try:
            # Load and preprocess image
            original_image = face_recognition.load_image_file(image_path)
            enhanced_image = self.enhance_image(original_image)
            
            # Multi-scale detection for better results
            face_infos = []
            all_debug_info = []
            
            # Primary detection with face_recognition
            locations_hog = face_recognition.face_locations(enhanced_image, model="hog")
            locations_cnn = []
            
            # Use CNN model for additional detection if enabled
            if self.model in ["cnn", "adaptive"]:
                try:
                    locations_cnn = face_recognition.face_locations(enhanced_image, model="cnn", number_of_times_to_upsample=1)
                except:
                    logging.warning("CNN model failed, falling back to HOG only")
            
            # Combine and deduplicate detections
            all_locations = self._merge_detections(locations_hog, locations_cnn)
            
            # Additional validation with OpenCV if available
            if self.face_cascade is not None:
                opencv_locations = self._opencv_detection(enhanced_image)
                all_locations = self._merge_detections(all_locations, opencv_locations)
            
            # Process each detected face
            for location in all_locations:
                top, right, bottom, left = location
                area = (right - left) * (bottom - top)
                
                # Apply minimum area filter early
                if min_face_area and area < min_face_area:
                    all_debug_info.append((location, area))
                    continue
                
                # Calculate quality metrics
                quality_metrics = self.calculate_face_quality(enhanced_image, location)
                
                # Get face encoding
                try:
                    encodings = face_recognition.face_encodings(enhanced_image, [location])
                    if not encodings:
                        continue
                    
                    encoding = encodings[0]
                    
                    # Calculate confidence based on encoding quality
                    confidence = self._calculate_encoding_confidence(encoding, quality_metrics)
                    
                    # Determine quality level
                    quality_level = self._determine_quality_level(quality_metrics, area, confidence)
                    
                    face_info = FaceInfo(
                        encoding=encoding,
                        location=location,
                        quality_score=quality_metrics['overall_quality'],
                        pose_score=quality_metrics['pose_score'],
                        sharpness_score=quality_metrics['sharpness'],
                        lighting_score=quality_metrics['lighting'],
                        area=area,
                        confidence=confidence,
                        quality_level=quality_level
                    )
                    
                    face_infos.append(face_info)
                    all_debug_info.append((location, area))
                    
                except Exception as e:
                    logging.warning(f"Failed to encode face: {e}")
                    continue
            
            return original_image, face_infos, all_debug_info
            
        except Exception as e:
            logging.error(f"Face detection failed for {image_path}: {e}")
            return None, [], []

    def _merge_detections(self, locations1: List, locations2: List, overlap_threshold: float = 0.5) -> List:
        """
        Merge face detections from different methods, removing duplicates.
        """
        if not locations2:
            return locations1
        if not locations1:
            return locations2
        
        merged = list(locations1)
        
        for loc2 in locations2:
            is_duplicate = False
            for loc1 in locations1:
                if self._calculate_overlap(loc1, loc2) > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(loc2)
        
        return merged

    def _calculate_overlap(self, loc1: Tuple, loc2: Tuple) -> float:
        """
        Calculate IoU (Intersection over Union) between two face locations.
        """
        top1, right1, bottom1, left1 = loc1
        top2, right2, bottom2, left2 = loc2
        
        # Calculate intersection
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)
        
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0
        
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        
        # Calculate union
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _opencv_detection(self, image: np.ndarray) -> List[Tuple]:
        """
        Additional face detection using OpenCV for validation.
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Convert OpenCV format to face_recognition format
        locations = []
        for (x, y, w, h) in faces:
            # Convert from (x, y, w, h) to (top, right, bottom, left)
            locations.append((y, x + w, y + h, x))
        
        return locations

    def _calculate_encoding_confidence(self, encoding: np.ndarray, quality_metrics: Dict) -> float:
        """
        Calculate confidence score for a face encoding based on various factors.
        """
        # Base confidence from encoding variance (higher variance = more distinctive features)
        encoding_variance = np.var(encoding)
        variance_score = min(1.0, encoding_variance / 0.1)  # Normalize
        
        # Quality-based confidence
        quality_score = quality_metrics['overall_quality']
        
        # Combine scores
        confidence = (variance_score * 0.4 + quality_score * 0.6)
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0

    def _determine_quality_level(self, quality_metrics: Dict, area: int, confidence: float) -> FaceQuality:
        """
        Determine the overall quality level of a detected face.
        """
        overall_quality = quality_metrics['overall_quality']
        sharpness = quality_metrics['sharpness']
        pose_score = quality_metrics['pose_score']
        
        # Scoring criteria
        if (overall_quality > 0.8 and sharpness > 80 and pose_score > 0.7 and 
            area > 5000 and confidence > 0.8):
            return FaceQuality.EXCELLENT
        elif (overall_quality > 0.6 and sharpness > 50 and pose_score > 0.5 and 
              area > 3000 and confidence > 0.6):
            return FaceQuality.GOOD
        elif (overall_quality > 0.4 and sharpness > 30 and pose_score > 0.3 and 
              area > 2000 and confidence > 0.4):
            return FaceQuality.FAIR
        else:
            return FaceQuality.POOR

    def filter_faces_by_quality(self, face_infos: List[FaceInfo], min_quality: FaceQuality = FaceQuality.FAIR) -> List[FaceInfo]:
        """
        Filter faces based on minimum quality requirements.
        """
        return [face for face in face_infos if face.quality_level.value >= min_quality.value]

    def get_quality_weighted_encodings(self, face_infos: List[FaceInfo]) -> List[Tuple[np.ndarray, float]]:
        """
        Get face encodings with quality weights for improved clustering.
        """
        return [(face.encoding, face.confidence * face.quality_score) for face in face_infos]