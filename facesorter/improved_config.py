import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ProcessingMode(Enum):
    SPEED = "speed"           # Fast processing, lower accuracy
    BALANCED = "balanced"     # Balance between speed and accuracy  
    ACCURACY = "accuracy"     # Highest accuracy, slower processing
    ADAPTIVE = "adaptive"     # Automatically adjust based on image quality

@dataclass
class FaceDetectionConfig:
    """Configuration for face detection parameters."""
    model: str = "adaptive"                    # hog, cnn, or adaptive
    enable_multi_detection: bool = True        # Use multiple detection methods
    enable_opencv_validation: bool = True      # Use OpenCV for additional validation
    min_face_area: int = 2000                 # Minimum face area in pixels
    max_faces_per_image: int = 20             # Maximum faces to process per image
    enable_image_enhancement: bool = True      # Apply image preprocessing
    quality_filtering: bool = True             # Filter faces by quality
    min_quality_level: str = "FAIR"           # Minimum quality level to keep

@dataclass  
class ClusteringConfig:
    """Configuration for face clustering parameters."""
    method: str = "hierarchical"              # hierarchical, dbscan, or adaptive
    enable_adaptive_thresholds: bool = True   # Use quality-based thresholds
    min_cluster_size: int = 1                 # Minimum faces per cluster
    enable_post_processing: bool = True       # Enable merge/split operations
    merge_similar_clusters: bool = True       # Automatically merge similar clusters
    split_diverse_clusters: bool = True       # Split overly diverse clusters
    confidence_threshold: float = 0.5         # Minimum confidence for clustering
    
    # Quality-specific thresholds
    excellent_threshold: float = 0.35
    good_threshold: float = 0.45  
    fair_threshold: float = 0.55
    poor_threshold: float = 0.65

@dataclass
class ProcessingConfig:
    """Configuration for processing workflow."""
    batch_size: int = 32                      # Batch size for parallel processing
    max_workers: int = 4                      # Maximum worker processes
    enable_gpu: bool = False                  # Use GPU acceleration if available
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    enable_progress_tracking: bool = True     # Show detailed progress
    enable_debug_output: bool = False         # Generate debug information

@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    enable_quality_scoring: bool = True       # Enable face quality assessment
    sharpness_weight: float = 0.4            # Weight for sharpness in quality score
    lighting_weight: float = 0.3             # Weight for lighting in quality score  
    pose_weight: float = 0.3                 # Weight for pose in quality score
    min_sharpness: float = 30.0              # Minimum sharpness threshold
    min_lighting_score: float = 20.0         # Minimum lighting score
    min_pose_score: float = 0.2              # Minimum pose score

class ImprovedConfig:
    """Enhanced configuration management with adaptive settings."""
    
    def __init__(self, config_file: str = "improved_config.yaml"):
        """Initialize configuration from file or defaults."""
        self.config_file = config_file
        self.face_detection = FaceDetectionConfig()
        self.clustering = ClusteringConfig()
        self.processing = ProcessingConfig()
        self.quality = QualityConfig()
        
        # Load from file if it exists
        if os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            # Create default config file
            self.save_to_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'face_detection' in data:
                self.face_detection = FaceDetectionConfig(**data['face_detection'])
            if 'clustering' in data:
                self.clustering = ClusteringConfig(**data['clustering'])  
            if 'processing' in data:
                # Handle enum conversion
                if 'processing_mode' in data['processing']:
                    data['processing']['processing_mode'] = ProcessingMode(data['processing']['processing_mode'])
                self.processing = ProcessingConfig(**data['processing'])
            if 'quality' in data:
                self.quality = QualityConfig(**data['quality'])
                
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
            print("Using default configuration.")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML file."""
        try:
            data = {
                'face_detection': asdict(self.face_detection),
                'clustering': asdict(self.clustering),
                'processing': asdict(self.processing),
                'quality': asdict(self.quality)
            }
            
            # Convert enum to string for YAML serialization
            data['processing']['processing_mode'] = self.processing.processing_mode.value
            
            with open(config_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save config to {config_file}: {e}")
    
    def get_adaptive_config(self, image_count: int, avg_image_size: Optional[float] = None) -> 'ImprovedConfig':
        """
        Get adaptive configuration based on dataset characteristics.
        
        Args:
            image_count: Number of images to process
            avg_image_size: Average image file size in MB
        """
        adapted_config = ImprovedConfig.__new__(ImprovedConfig)
        adapted_config.face_detection = FaceDetectionConfig(**asdict(self.face_detection))
        adapted_config.clustering = ClusteringConfig(**asdict(self.clustering))
        adapted_config.processing = ProcessingConfig(**asdict(self.processing))
        adapted_config.quality = QualityConfig(**asdict(self.quality))
        
        # Adapt based on image count
        if image_count < 50:
            # Small dataset - prioritize accuracy
            adapted_config.face_detection.model = "cnn"
            adapted_config.clustering.method = "hierarchical"
            adapted_config.processing.processing_mode = ProcessingMode.ACCURACY
            adapted_config.processing.max_workers = min(4, os.cpu_count() or 1)
            
        elif image_count < 500:
            # Medium dataset - balanced approach
            adapted_config.face_detection.model = "adaptive"
            adapted_config.clustering.method = "hierarchical"
            adapted_config.processing.processing_mode = ProcessingMode.BALANCED
            adapted_config.processing.max_workers = min(6, os.cpu_count() or 1)
            
        else:
            # Large dataset - prioritize speed
            adapted_config.face_detection.model = "hog"
            adapted_config.clustering.method = "dbscan"
            adapted_config.processing.processing_mode = ProcessingMode.SPEED
            adapted_config.processing.max_workers = min(8, os.cpu_count() or 1)
            adapted_config.processing.batch_size = 64
        
        # Adapt based on image size
        if avg_image_size and avg_image_size > 5.0:  # Large images (>5MB)
            adapted_config.processing.batch_size = max(8, adapted_config.processing.batch_size // 2)
            adapted_config.face_detection.max_faces_per_image = 15
        
        return adapted_config
    
    def get_quality_based_thresholds(self) -> Dict[str, float]:
        """Get quality-based clustering thresholds."""
        return {
            'EXCELLENT': self.clustering.excellent_threshold,
            'GOOD': self.clustering.good_threshold,
            'FAIR': self.clustering.fair_threshold,
            'POOR': self.clustering.poor_threshold
        }
    
    def update_from_performance_feedback(self, false_positives: int, false_negatives: int, total_faces: int) -> None:
        """
        Update configuration based on performance feedback.
        
        Args:
            false_positives: Number of incorrectly grouped faces
            false_negatives: Number of faces that should have been grouped but weren't
            total_faces: Total number of faces processed
        """
        if total_faces == 0:
            return
            
        fp_rate = false_positives / total_faces
        fn_rate = false_negatives / total_faces
        
        # If too many false positives (same person split into multiple groups)
        if fp_rate > 0.1:  # More than 10% false positives
            # Make clustering more lenient
            self.clustering.excellent_threshold = min(0.45, self.clustering.excellent_threshold + 0.05)
            self.clustering.good_threshold = min(0.55, self.clustering.good_threshold + 0.05)
            self.clustering.fair_threshold = min(0.65, self.clustering.fair_threshold + 0.05)
            self.clustering.poor_threshold = min(0.75, self.clustering.poor_threshold + 0.05)
            
        # If too many false negatives (different people grouped together)  
        elif fn_rate > 0.1:  # More than 10% false negatives
            # Make clustering more strict
            self.clustering.excellent_threshold = max(0.25, self.clustering.excellent_threshold - 0.05)
            self.clustering.good_threshold = max(0.35, self.clustering.good_threshold - 0.05)
            self.clustering.fair_threshold = max(0.45, self.clustering.fair_threshold - 0.05)
            self.clustering.poor_threshold = max(0.55, self.clustering.poor_threshold - 0.05)
        
        # Save updated configuration
        self.save_to_file(self.config_file)
    
    def get_processing_recommendations(self) -> Dict[str, Any]:
        """Get processing recommendations based on current configuration."""
        recommendations = {
            'estimated_speed': 'medium',
            'estimated_accuracy': 'medium', 
            'memory_usage': 'medium',
            'cpu_usage': 'medium',
            'recommendations': []
        }
        
        # Analyze current settings
        if self.face_detection.model == "cnn":
            recommendations['estimated_accuracy'] = 'high'
            recommendations['estimated_speed'] = 'slow'
            recommendations['cpu_usage'] = 'high'
            if not self.processing.enable_gpu:
                recommendations['recommendations'].append(
                    "Consider enabling GPU acceleration for CNN model"
                )
        
        if self.processing.max_workers > (os.cpu_count() or 4):
            recommendations['recommendations'].append(
                f"Max workers ({self.processing.max_workers}) exceeds CPU count. Consider reducing."
            )
        
        if self.clustering.method == "hierarchical" and self.clustering.enable_post_processing:
            recommendations['estimated_accuracy'] = 'high'
            recommendations['estimated_speed'] = 'slow'
            
        if not self.face_detection.quality_filtering:
            recommendations['recommendations'].append(
                "Enable quality filtering to improve clustering accuracy"
            )
            
        return recommendations

# Global configuration instance
improved_config = ImprovedConfig()

# Legacy compatibility
def get_config_value(key: str, default: Any = None) -> Any:
    """Get configuration value with dot notation for backward compatibility."""
    try:
        parts = key.split('.')
        obj = improved_config
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
                
        return obj
    except:
        return default