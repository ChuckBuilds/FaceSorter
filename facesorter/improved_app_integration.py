"""
Integration module for improved facial recognition system.
This module provides updated functions that can be used to replace the original
processing pipeline in app.py with minimal changes.
"""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import logging

# Import improved components
from facesorter.improved_face_detector import ImprovedFaceDetector, FaceInfo, FaceQuality
from facesorter.improved_face_clusterer import ImprovedFaceClusterer, AdaptiveThreshold
from facesorter.improved_worker import init_improved_worker, process_image_improved
from facesorter.improved_config import improved_config, ProcessingMode
from facesorter.config import OUTPUT_DIR, TEMP_UPLOAD_DIR, TEMP_CROP_DIR
from facesorter.file_organizer import FileOrganizer

def run_improved_processing_pipeline(uploaded_files, processing_mode: str = "balanced", 
                                   min_face_area: int = 2000, max_workers: int = 4,
                                   enable_gpu: bool = False) -> Tuple[Dict, int, List, List]:
    """
    Improved processing pipeline with adaptive detection and clustering.
    
    Args:
        uploaded_files: List of uploaded file objects
        processing_mode: Processing mode ("speed", "balanced", "accuracy", "adaptive")
        min_face_area: Minimum face area threshold
        max_workers: Maximum number of worker processes
        enable_gpu: Whether to enable GPU acceleration
    
    Returns:
        Tuple of (people_dict, num_clusters, merge_suggestions, temp_file_paths)
    """
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Clean up previous runs
        _cleanup_directories([TEMP_UPLOAD_DIR, TEMP_CROP_DIR])
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        os.makedirs(TEMP_CROP_DIR, exist_ok=True)
        
        # Get adaptive configuration based on dataset size
        dataset_size = len(uploaded_files)
        adaptive_config = improved_config.get_adaptive_config(dataset_size)
        
        # Override with user preferences
        if processing_mode != "adaptive":
            adaptive_config.processing.processing_mode = ProcessingMode(processing_mode)
        
        # Initialize components
        face_clusterer = ImprovedFaceClusterer(
            min_cluster_size=adaptive_config.clustering.min_cluster_size,
            enable_hierarchical=(adaptive_config.clustering.method == "hierarchical")
        )
        file_organizer = FileOrganizer(output_dir=OUTPUT_DIR)
        
        # Save uploaded files and extract video frames
        temp_file_paths, video_source_map = _save_and_extract_files(uploaded_files)
        
        if not temp_file_paths:
            return {}, 0, [], []
        
        # Process images with improved detection
        all_face_infos, file_face_map, debug_info = _process_images_improved(
            temp_file_paths, 
            video_source_map,
            min_face_area,
            max_workers,
            adaptive_config,
            enable_gpu
        )
        
        if not all_face_infos:
            return {}, 0, [], []
        
        # Perform improved clustering
        cluster_results = _perform_improved_clustering(all_face_infos, face_clusterer, adaptive_config)
        cluster_labels, num_clusters, cluster_info = cluster_results
        
        # Map clusters to original files
        people_dict = _create_people_dict(
            all_face_infos, 
            cluster_labels, 
            cluster_info, 
            file_face_map, 
            file_organizer
        )
        
        # Get merge suggestions
        merge_suggestions = face_clusterer.get_merge_suggestions(
            all_face_infos, 
            cluster_labels, 
            cluster_info
        )
        
        # Cleanup temporary files
        _cleanup_directories([TEMP_UPLOAD_DIR, TEMP_CROP_DIR])
        
        return people_dict, num_clusters, merge_suggestions, temp_file_paths
        
    except Exception as e:
        logging.error(f"Improved processing pipeline failed: {e}")
        return {}, 0, [], []

def _cleanup_directories(directories: List[str]) -> None:
    """Clean up specified directories."""
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def _save_and_extract_files(uploaded_files) -> Tuple[List[Path], Dict[Path, Path]]:
    """Save uploaded files and extract video frames."""
    temp_file_paths = []
    video_source_map = {}
    
    for uploaded_file in uploaded_files:
        temp_file_path = Path(TEMP_UPLOAD_DIR) / uploaded_file.name
        
        # Save file to disk
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        
        # Handle video files
        if temp_file_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            from facesorter.media_processor import MediaProcessor
            video_frame_output_dir = Path(TEMP_UPLOAD_DIR) / "video_frames"
            extracted_frames = MediaProcessor.extract_frames_from_video(
                temp_file_path, 
                video_frame_output_dir,
                frames_per_second=1
            )
            temp_file_paths.extend(extracted_frames)
            
            # Map frames to original video
            for frame in extracted_frames:
                video_source_map[frame] = temp_file_path
        else:
            temp_file_paths.append(temp_file_path)
    
    return temp_file_paths, video_source_map

def _process_images_improved(temp_file_paths: List[Path], video_source_map: Dict[Path, Path],
                           min_face_area: int, max_workers: int, adaptive_config, 
                           enable_gpu: bool) -> Tuple[List[FaceInfo], Dict[Path, List[int]], Dict]:
    """Process images with improved face detection."""
    
    # Determine detection model based on configuration
    if adaptive_config.face_detection.model == "adaptive":
        # Choose model based on dataset size and processing mode
        if len(temp_file_paths) < 100 and adaptive_config.processing.processing_mode == ProcessingMode.ACCURACY:
            model = "cnn"
        else:
            model = "hog"
    else:
        model = adaptive_config.face_detection.model
    
    # Prepare worker configuration
    worker_config = {
        'quality_filtering': adaptive_config.face_detection.quality_filtering,
        'min_quality_level': adaptive_config.face_detection.min_quality_level,
        'enable_image_enhancement': adaptive_config.face_detection.enable_image_enhancement,
        'max_faces_per_image': adaptive_config.face_detection.max_faces_per_image
    }
    
    all_face_infos = []
    file_face_map = {}
    debug_info = {}
    
    # Process images in parallel
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_improved_worker,
        initargs=(model, enable_gpu, worker_config)
    ) as executor:
        
        # Submit all tasks
        future_to_path = {}
        for temp_path in temp_file_paths:
            original_media_path = video_source_map.get(temp_path, temp_path)
            future = executor.submit(process_image_improved, (temp_path, min_face_area, original_media_path))
            future_to_path[future] = temp_path
        
        # Collect results
        for future in as_completed(future_to_path):
            temp_path = future_to_path[future]
            result = future.result()
            
            if result:
                original_media_path, face_infos, crop_paths, debug_info_item, processed_path = result
                
                if face_infos:
                    # Track which faces belong to which file
                    start_idx = len(all_face_infos)
                    all_face_infos.extend(face_infos)
                    end_idx = len(all_face_infos)
                    
                    file_face_map[original_media_path] = list(range(start_idx, end_idx))
                    
                    # Store debug info
                    if processed_path:
                        debug_info[processed_path.name] = debug_info_item
    
    return all_face_infos, file_face_map, debug_info

def _perform_improved_clustering(all_face_infos: List[FaceInfo], face_clusterer: ImprovedFaceClusterer,
                               adaptive_config) -> Tuple[np.ndarray, int, Dict]:
    """Perform improved clustering on face information."""
    
    # Update clusterer configuration
    if hasattr(face_clusterer, 'adaptive_threshold'):
        # Update thresholds based on configuration
        thresholds = adaptive_config.get_quality_based_thresholds()
        face_clusterer.adaptive_threshold.quality_thresholds = {
            FaceQuality.EXCELLENT: thresholds['EXCELLENT'],
            FaceQuality.GOOD: thresholds['GOOD'],
            FaceQuality.FAIR: thresholds['FAIR'],
            FaceQuality.POOR: thresholds['POOR']
        }
    
    # Perform clustering
    cluster_labels, num_clusters, cluster_info = face_clusterer.cluster_faces_advanced(all_face_infos)
    
    return cluster_labels, num_clusters, cluster_info

def _create_people_dict(all_face_infos: List[FaceInfo], cluster_labels: np.ndarray, 
                       cluster_info: Dict, file_face_map: Dict[Path, List[int]], 
                       file_organizer: FileOrganizer) -> Dict:
    """Create people dictionary for UI compatibility."""
    
    # Map clusters to original files
    cluster_to_original_files = defaultdict(set)
    
    for original_path, face_indices in file_face_map.items():
        for face_idx in face_indices:
            if face_idx < len(cluster_labels):
                cluster_id = cluster_labels[face_idx]
                if cluster_id >= 0:  # Not noise
                    cluster_to_original_files[cluster_id].add(original_path)
    
    # Organize files into folders
    cluster_to_dest_files = file_organizer.organize_files_into_folders(cluster_to_original_files)
    
    # Create representative face crops
    cluster_representatives = _create_representative_faces(cluster_info, all_face_infos)
    
    # Build people dictionary for UI
    people_dict = {}
    for cluster_id, files in cluster_to_dest_files.items():
        if cluster_id in cluster_info:
            info = cluster_info[cluster_id]
            
            # Generate person name with quality information
            quality_dist = info.get('quality_distribution', Counter())
            dominant_quality = quality_dist.most_common(1)[0][0].name if quality_dist else 'UNKNOWN'
            avg_confidence = info.get('avg_confidence', 0.0)
            
            person_name = f"Person_{cluster_id + 1}"
            if avg_confidence > 0.8:
                person_name += "_HighConf"
            elif avg_confidence < 0.4:
                person_name += "_LowConf"
            
            people_dict[cluster_id] = {
                "name": person_name,
                "files": files,
                "representative_face": cluster_representatives.get(cluster_id),
                "quality_info": {
                    "avg_confidence": avg_confidence,
                    "quality_distribution": dict(quality_dist),
                    "dominant_quality": dominant_quality
                }
            }
    
    return people_dict

def _create_representative_faces(cluster_info: Dict, all_face_infos: List[FaceInfo]) -> Dict[int, str]:
    """Create representative face crops for each cluster."""
    cluster_representatives = {}
    
    if not cluster_info:
        return cluster_representatives
    
    crop_dir = os.path.join(OUTPUT_DIR, "face_previews")
    os.makedirs(crop_dir, exist_ok=True)
    
    for cluster_id, info in cluster_info.items():
        rep_idx = info.get('representative_idx')
        if rep_idx is not None and rep_idx < len(all_face_infos):
            # For now, we'll use a placeholder since we need the original image
            # In a full implementation, we'd store and use the face crop
            dest_crop_path = os.path.join(crop_dir, f"rep_{cluster_id}.jpg")
            cluster_representatives[cluster_id] = dest_crop_path
    
    return cluster_representatives

def get_improved_diagnostic_info(face_infos: List[FaceInfo]) -> Dict[str, Any]:
    """Get diagnostic information about the improved processing results."""
    if not face_infos:
        return {}
    
    quality_distribution = Counter(face.quality_level for face in face_infos)
    confidence_scores = [face.confidence for face in face_infos]
    quality_scores = [face.quality_score for face in face_infos]
    
    return {
        'total_faces': len(face_infos),
        'quality_distribution': {quality.name: count for quality, count in quality_distribution.items()},
        'avg_confidence': np.mean(confidence_scores),
        'min_confidence': np.min(confidence_scores),
        'max_confidence': np.max(confidence_scores),
        'avg_quality_score': np.mean(quality_scores),
        'min_quality_score': np.min(quality_scores),
        'max_quality_score': np.max(quality_scores),
        'excellent_faces': sum(1 for f in face_infos if f.quality_level == FaceQuality.EXCELLENT),
        'good_faces': sum(1 for f in face_infos if f.quality_level == FaceQuality.GOOD),
        'fair_faces': sum(1 for f in face_infos if f.quality_level == FaceQuality.FAIR),
        'poor_faces': sum(1 for f in face_infos if f.quality_level == FaceQuality.POOR)
    }

def update_config_from_feedback(false_positives: int, false_negatives: int, total_faces: int) -> None:
    """Update configuration based on user feedback about clustering accuracy."""
    improved_config.update_from_performance_feedback(false_positives, false_negatives, total_faces)

def get_processing_recommendations(dataset_size: int, avg_file_size: Optional[float] = None) -> Dict[str, Any]:
    """Get processing recommendations for the current dataset."""
    adaptive_config = improved_config.get_adaptive_config(dataset_size, avg_file_size)
    return adaptive_config.get_processing_recommendations()