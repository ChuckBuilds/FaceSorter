import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging

# Make sure all custom modules are in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facesorter.improved_face_detector import ImprovedFaceDetector, FaceInfo, FaceQuality
from facesorter.improved_config import improved_config
from facesorter.config import TEMP_CROP_DIR, TEMP_UPLOAD_DIR

# Global variables for worker processes
improved_face_detector = None
worker_config = None

def init_improved_worker(model: str = "adaptive", enable_gpu: bool = False, config_dict: Optional[Dict] = None):
    """
    Initialize worker process with improved face detector and configuration.
    
    Args:
        model: Face detection model to use
        enable_gpu: Whether to enable GPU acceleration
        config_dict: Configuration dictionary for the worker
    """
    global improved_face_detector, worker_config
    
    # Initialize the improved face detector
    improved_face_detector = ImprovedFaceDetector(model=model, enable_gpu=enable_gpu)
    
    # Set worker configuration
    if config_dict:
        worker_config = config_dict
    else:
        worker_config = {
            'quality_filtering': improved_config.face_detection.quality_filtering,
            'min_quality_level': improved_config.face_detection.min_quality_level,
            'enable_image_enhancement': improved_config.face_detection.enable_image_enhancement,
            'max_faces_per_image': improved_config.face_detection.max_faces_per_image
        }
    
    logging.info(f"Improved worker initialized with model: {model}, GPU: {enable_gpu}")

def process_image_improved(args: Tuple) -> Optional[Tuple]:
    """
    Improved image processing function with quality assessment and adaptive detection.
    
    Args:
        args: Tuple containing (temp_file_path, min_face_area, original_media_path, processing_options)
    
    Returns:
        Tuple of (original_media_path, face_infos, crop_paths, debug_info, temp_file_path) or None
    """
    global improved_face_detector, worker_config
    
    if len(args) == 3:
        temp_file_path, min_face_area, original_media_path = args
        processing_options = {}
    else:
        temp_file_path, min_face_area, original_media_path, processing_options = args
    
    try:
        if not temp_file_path.exists():
            return None
        
        # Process based on file type
        if temp_file_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            return _process_video_improved(temp_file_path, min_face_area, original_media_path, processing_options)
        else:
            return _process_image_file_improved(temp_file_path, min_face_area, original_media_path, processing_options)
            
    except Exception as e:
        logging.error(f"Improved worker failed on {temp_file_path.name}: {e}")
        return None

def _process_image_file_improved(temp_file_path: Path, min_face_area: int, 
                                original_media_path: Path, processing_options: Dict) -> Optional[Tuple]:
    """Process a single image file with improved detection."""
    global improved_face_detector, worker_config
    
    try:
        # Use improved face detection
        original_image, face_infos, debug_info = improved_face_detector.detect_faces_adaptive(
            str(temp_file_path), 
            min_face_area=min_face_area
        )
        
        if original_image is None or not face_infos:
            return None
        
        # Apply quality filtering if enabled
        if worker_config.get('quality_filtering', True):
            min_quality_str = worker_config.get('min_quality_level', 'FAIR')
            min_quality = FaceQuality[min_quality_str]
            face_infos = improved_face_detector.filter_faces_by_quality(face_infos, min_quality)
        
        # Limit number of faces per image
        max_faces = worker_config.get('max_faces_per_image', 20)
        if len(face_infos) > max_faces:
            # Sort by quality score and keep the best ones
            face_infos = sorted(face_infos, key=lambda f: f.quality_score * f.confidence, reverse=True)[:max_faces]
        
        if not face_infos:
            return None
        
        # Generate face crops
        crop_paths = []
        for i, face_info in enumerate(face_infos):
            crop_path = _save_face_crop(original_image, face_info, temp_file_path, i)
            if crop_path:
                crop_paths.append(crop_path)
        
        # Prepare debug information for UI
        debug_info_for_ui = {
            "face_locations": debug_info,
            "quality_info": [
                {
                    "quality_level": face_info.quality_level.name,
                    "quality_score": face_info.quality_score,
                    "confidence": face_info.confidence,
                    "pose_score": face_info.pose_score,
                    "sharpness": face_info.sharpness_score,
                    "lighting": face_info.lighting_score,
                    "area": face_info.area
                }
                for face_info in face_infos
            ]
        }
        
        return (original_media_path, face_infos, crop_paths, debug_info_for_ui, temp_file_path)
        
    except Exception as e:
        logging.error(f"Failed to process image {temp_file_path.name}: {e}")
        return None

def _process_video_improved(temp_file_path: Path, min_face_area: int, 
                           original_media_path: Path, processing_options: Dict) -> Optional[Tuple]:
    """Process video file with improved frame extraction and face detection."""
    global improved_face_detector, worker_config
    
    try:
        import cv2
        from PIL import Image
        
        # Extract frames from video
        cap = cv2.VideoCapture(str(temp_file_path))
        if not cap.isOpened():
            return None
        
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            frame_rate = 30  # Default fallback
        
        # Extract frames at intervals (e.g., 1 frame per second)
        frame_interval = max(1, int(frame_rate))
        max_frames = processing_options.get('max_video_frames', 10)
        
        all_face_infos = []
        all_crop_paths = []
        all_debug_info = []
        
        frame_number = 0
        processed_frames = 0
        
        temp_dir = Path(TEMP_UPLOAD_DIR)
        temp_dir.mkdir(exist_ok=True)
        
        while cap.isOpened() and processed_frames < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame temporarily
            temp_frame_path = temp_dir / f"{temp_file_path.stem}_frame_{frame_number}.jpg"
            Image.fromarray(rgb_frame).save(temp_frame_path, "JPEG", quality=95)
            
            try:
                # Process frame with improved detection
                frame_image, frame_face_infos, frame_debug_info = improved_face_detector.detect_faces_adaptive(
                    str(temp_frame_path),
                    min_face_area=min_face_area
                )
                
                if frame_face_infos:
                    # Apply quality filtering
                    if worker_config.get('quality_filtering', True):
                        min_quality_str = worker_config.get('min_quality_level', 'FAIR')
                        min_quality = FaceQuality[min_quality_str]
                        frame_face_infos = improved_face_detector.filter_faces_by_quality(frame_face_infos, min_quality)
                    
                    # Generate crops for this frame
                    for i, face_info in enumerate(frame_face_infos):
                        crop_path = _save_face_crop(frame_image, face_info, temp_frame_path, i)
                        if crop_path:
                            all_crop_paths.append(crop_path)
                    
                    all_face_infos.extend(frame_face_infos)
                    all_debug_info.extend(frame_debug_info)
                
            finally:
                # Clean up temporary frame file
                if temp_frame_path.exists():
                    temp_frame_path.unlink()
            
            frame_number += frame_interval
            processed_frames += 1
        
        cap.release()
        
        if not all_face_infos:
            return None
        
        # Prepare debug information
        debug_info_for_ui = {
            "face_locations": all_debug_info,
            "frames_processed": processed_frames,
            "quality_info": [
                {
                    "quality_level": face_info.quality_level.name,
                    "quality_score": face_info.quality_score,
                    "confidence": face_info.confidence,
                    "pose_score": face_info.pose_score,
                    "sharpness": face_info.sharpness_score,
                    "lighting": face_info.lighting_score,
                    "area": face_info.area
                }
                for face_info in all_face_infos
            ]
        }
        
        return (original_media_path, all_face_infos, all_crop_paths, debug_info_for_ui, temp_file_path)
        
    except Exception as e:
        logging.error(f"Failed to process video {temp_file_path.name}: {e}")
        return None

def _save_face_crop(image: np.ndarray, face_info: FaceInfo, source_path: Path, face_index: int) -> Optional[Path]:
    """Save a face crop to disk with quality information in filename."""
    try:
        from PIL import Image
        
        # Create crop directory if it doesn't exist
        os.makedirs(TEMP_CROP_DIR, exist_ok=True)
        
        # Extract face region with padding
        top, right, bottom, left = face_info.location
        padding = 20  # Add some padding around the face
        
        height, width = image.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        
        # Crop the face
        face_crop = image[top:bottom, left:right]
        
        if face_crop.size == 0:
            return None
        
        # Generate filename with quality information
        quality_level = face_info.quality_level.name.lower()
        confidence_str = f"{int(face_info.confidence * 100):02d}"
        crop_filename = f"{source_path.stem}_{face_index}_{quality_level}_{confidence_str}.jpg"
        crop_path = Path(TEMP_CROP_DIR) / crop_filename
        
        # Save as PIL Image
        pil_image = Image.fromarray(face_crop)
        pil_image.save(crop_path, "JPEG", quality=95)
        
        return crop_path
        
    except Exception as e:
        logging.error(f"Failed to save face crop: {e}")
        return None

def get_worker_statistics() -> Dict[str, Any]:
    """Get statistics about the current worker configuration."""
    global worker_config, improved_face_detector
    
    stats = {
        'worker_initialized': improved_face_detector is not None,
        'config': worker_config.copy() if worker_config else {},
        'detector_model': getattr(improved_face_detector, 'model', 'unknown') if improved_face_detector else 'none'
    }
    
    return stats

# Backward compatibility function
def _process_image_worker(args):
    """Backward compatibility wrapper for the original worker function."""
    return process_image_improved(args)