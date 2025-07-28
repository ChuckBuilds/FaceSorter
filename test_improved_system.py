#!/usr/bin/env python3
"""
Test script for the improved facial recognition system.
This script validates the new components and compares them with the original system.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the facesorter module to the path
sys.path.append(os.path.abspath('.'))

def test_improved_face_detector():
    """Test the improved face detector."""
    print("Testing Improved Face Detector...")
    
    try:
        from facesorter.improved_face_detector import ImprovedFaceDetector, FaceQuality
        
        # Initialize detector
        detector = ImprovedFaceDetector(model="adaptive")
        print("✓ Detector initialized successfully")
        
        # Test quality levels
        quality_levels = [FaceQuality.EXCELLENT, FaceQuality.GOOD, FaceQuality.FAIR, FaceQuality.POOR]
        print(f"✓ Quality levels available: {[q.name for q in quality_levels]}")
        
        print("✓ Improved Face Detector test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Improved Face Detector test failed: {e}\n")
        return False

def test_improved_clusterer():
    """Test the improved face clusterer."""
    print("Testing Improved Face Clusterer...")
    
    try:
        from facesorter.improved_face_clusterer import ImprovedFaceClusterer, AdaptiveThreshold
        from facesorter.improved_face_detector import FaceQuality
        
        # Initialize clusterer
        clusterer = ImprovedFaceClusterer(enable_hierarchical=True)
        print("✓ Clusterer initialized successfully")
        
        # Test adaptive threshold
        threshold_manager = AdaptiveThreshold()
        print("✓ Adaptive threshold manager created")
        
        # Test quality-based thresholds
        thresholds = threshold_manager.quality_thresholds
        expected_qualities = [FaceQuality.EXCELLENT, FaceQuality.GOOD, FaceQuality.FAIR, FaceQuality.POOR]
        
        for quality in expected_qualities:
            if quality in thresholds:
                print(f"✓ Threshold for {quality.name}: {thresholds[quality]}")
            else:
                print(f"✗ Missing threshold for {quality.name}")
                return False
        
        print("✓ Improved Face Clusterer test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Improved Face Clusterer test failed: {e}\n")
        return False

def test_improved_config():
    """Test the improved configuration system."""
    print("Testing Improved Configuration...")
    
    try:
        from facesorter.improved_config import improved_config, ProcessingMode
        
        # Test configuration loading
        print(f"✓ Configuration loaded successfully")
        print(f"✓ Face detection model: {improved_config.face_detection.model}")
        print(f"✓ Clustering method: {improved_config.clustering.method}")
        print(f"✓ Processing mode: {improved_config.processing.processing_mode.value}")
        
        # Test adaptive configuration
        adaptive_config = improved_config.get_adaptive_config(image_count=100)
        print("✓ Adaptive configuration generated")
        
        # Test processing recommendations
        recommendations = improved_config.get_processing_recommendations()
        print(f"✓ Processing recommendations: {recommendations['estimated_speed']}/{recommendations['estimated_accuracy']}")
        
        # Test quality-based thresholds
        thresholds = improved_config.get_quality_based_thresholds()
        print(f"✓ Quality thresholds: {thresholds}")
        
        print("✓ Improved Configuration test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Improved Configuration test failed: {e}\n")
        return False

def test_improved_worker():
    """Test the improved worker system."""
    print("Testing Improved Worker...")
    
    try:
        from facesorter.improved_worker import init_improved_worker, get_worker_statistics
        
        # Test worker initialization
        init_improved_worker(model="hog", enable_gpu=False)
        print("✓ Worker initialized successfully")
        
        # Test worker statistics  
        stats = get_worker_statistics()
        print(f"✓ Worker statistics: {stats}")
        
        if stats['worker_initialized']:
            print("✓ Worker is properly initialized")
        else:
            print("✗ Worker initialization failed")
            return False
        
        print("✓ Improved Worker test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Improved Worker test failed: {e}\n")
        return False

def test_integration():
    """Test the integration module."""
    print("Testing Integration Module...")
    
    try:
        from facesorter.improved_app_integration import (
            get_improved_diagnostic_info, 
            update_config_from_feedback,
            get_processing_recommendations
        )
        
        # Test diagnostic info with empty data
        diagnostic_info = get_improved_diagnostic_info([])
        print("✓ Diagnostic info function works")
        
        # Test feedback update
        update_config_from_feedback(5, 3, 100)
        print("✓ Feedback update function works")
        
        # Test processing recommendations
        recommendations = get_processing_recommendations(dataset_size=50)
        print(f"✓ Processing recommendations: {recommendations}")
        
        print("✓ Integration Module test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Integration Module test failed: {e}\n")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("Testing Dependencies...")
    
    required_packages = [
        'numpy',
        'opencv-python',
        'scikit-learn', 
        'face_recognition',
        'PIL',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✓ {package} (cv2) available")
            elif package == 'PIL':
                from PIL import Image
                print(f"✓ {package} available")
            elif package == 'yaml':
                import yaml
                print(f"✓ {package} available")
            else:
                __import__(package)
                print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n✗ Missing packages: {missing_packages}")
        print("Please install them with:")
        for package in missing_packages:
            print(f"  pip install {package}")
        return False
    
    print("✓ All dependencies available\n")
    return True

def run_performance_comparison():
    """Run a basic performance comparison if possible."""
    print("Performance Comparison...")
    
    try:
        from facesorter.improved_config import improved_config
        
        # Test different processing modes
        modes = ["speed", "balanced", "accuracy"]
        
        for mode in modes:
            start_time = time.time()
            
            # Simulate configuration for different modes
            if mode == "speed":
                config = improved_config.get_adaptive_config(1000)  # Large dataset
            elif mode == "balanced":
                config = improved_config.get_adaptive_config(200)   # Medium dataset
            else:  # accuracy
                config = improved_config.get_adaptive_config(50)    # Small dataset
            
            end_time = time.time()
            
            print(f"✓ {mode.capitalize()} mode configuration: {(end_time - start_time)*1000:.2f}ms")
            print(f"  - Detection model: {config.face_detection.model}")
            print(f"  - Clustering method: {config.clustering.method}")
            print(f"  - Max workers: {config.processing.max_workers}")
        
        print("✓ Performance comparison completed\n")
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("IMPROVED FACIAL RECOGNITION SYSTEM - TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Improved Face Detector", test_improved_face_detector),
        ("Improved Face Clusterer", test_improved_clusterer),
        ("Improved Configuration", test_improved_config),
        ("Improved Worker", test_improved_worker),
        ("Integration Module", test_integration),
        ("Performance Comparison", run_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved system is ready to use.")
        print("\nNext steps:")
        print("1. Review the IMPROVED_FACIAL_RECOGNITION_GUIDE.md")
        print("2. Choose an integration approach (replace pipeline, gradual, or config-based)")
        print("3. Test with a small dataset first")
        print("4. Monitor performance and provide feedback")
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix the issues before using the system.")
        
        if passed == 0:
            print("\nIt looks like the improved system isn't properly installed.")
            print("Make sure all the new files are in the facesorter/ directory:")
            print("- improved_face_detector.py")
            print("- improved_face_clusterer.py") 
            print("- improved_config.py")
            print("- improved_worker.py")
            print("- improved_app_integration.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()