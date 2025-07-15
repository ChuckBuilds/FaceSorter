import os
from pathlib import Path
import face_recognition
from PIL import Image
import cv2

# Make sure all custom modules are in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facesorter.face_detector import FaceDetector
from facesorter.config import TEMP_CROP_DIR, TEMP_UPLOAD_DIR

# --- Worker Process Initialization ---

# This global variable will hold the FaceDetector instance for a single worker process.
face_detector = None

def init_worker(model):
    """
    Initializer for each worker process.
    Creates a single FaceDetector instance for the life of the process.
    """
    global face_detector
    face_detector = FaceDetector(model=model)

def _process_image_worker(args):
    """
    Helper function for multiprocessing. It detects and encodes faces in a single image.
    This must be a top-level function to be pickle-able by multiprocessing.
    """
    temp_file_path, min_face_area, original_media_path = args
    global face_detector
    try:
        if not temp_file_path.exists():
            return None # Skip if the source file is gone

        img_path = temp_file_path # For images, the path is direct
        
        encodings = []
        crops = []
        debug_info_from_detector = []

        if temp_file_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
            # Video file processing
            all_frame_encodings = []
            all_frame_crops = []
            all_frame_debug_info = []
            
            # This needs to be fetched from session state if it exists, but workers don't have access.
            # A cleaner approach would be to pass it in `args`, but for now, we assume a default.
            temp_dir = Path(TEMP_UPLOAD_DIR)
            
            cap = cv2.VideoCapture(str(temp_file_path))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            if frame_rate == 0:
                cap.release()
                return None
            
            frame_interval = int(frame_rate) # Process one frame per second
            
            frame_number = 0
            processed_frames = 0

            while cap.isOpened() and processed_frames < 10: # Limit to 10 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Convert the frame to RGB
                    rgb_frame = frame[:, :, ::-1]
                    
                    # Use a temporary file to work with face_recognition library
                    temp_frame_path_for_worker = temp_dir / f"{temp_file_path.stem}_frame_{frame_number}.jpg"
                    Image.fromarray(rgb_frame).save(temp_frame_path_for_worker)
                    
                    image, frame_encodings, frame_locations, frame_debug_info = face_detector.detect_faces(temp_frame_path_for_worker, min_face_area=min_face_area)
                    
                    if frame_encodings:
                        all_frame_encodings.extend(frame_encodings)
                        # Now pass the returned image and locations to crop_faces
                        if image is not None:
                            frame_crops = face_detector.crop_faces(image, frame_locations)
                            all_frame_crops.extend(frame_crops)

                    if frame_debug_info:
                        all_frame_debug_info.extend(frame_debug_info)

                    # Clean up the temp frame file
                    if temp_frame_path_for_worker.exists():
                        temp_frame_path_for_worker.unlink()
                        
                    processed_frames += 1
                else:
                    break
                
                frame_number += frame_interval

            cap.release()
            
            encodings = all_frame_encodings
            crops = all_frame_crops
            debug_info_from_detector = all_frame_debug_info
        else:
            # Image file processing
            image, encodings, face_locations, debug_info_from_detector = face_detector.detect_faces(img_path, min_face_area=min_face_area)
            if encodings and image is not None:
                crops = face_detector.crop_faces(image, face_locations)

        # This is the debug info for the UI.
        debug_info_for_ui = { "face_locations": debug_info_from_detector }

        if encodings:
            # Save crops to a temporary directory and return paths
            crop_paths = []
            for i, crop_img in enumerate(crops):
                # Ensure the temp crop dir exists
                os.makedirs(TEMP_CROP_DIR, exist_ok=True)
                crop_filename = f"{temp_file_path.stem}_{i}.jpg"
                crop_path = os.path.join(TEMP_CROP_DIR, crop_filename)
                crop_img.save(crop_path, "JPEG")
                crop_paths.append(crop_path)
            
            # Return the original source media path, encodings, crop paths, and the processed path
            return (original_media_path, encodings, crop_paths, debug_info_for_ui, temp_file_path)
    except Exception as e:
        # This will print to the console where streamlit is running
        print(f"Worker process failed on {os.path.basename(temp_file_path)}: {e}")
    return None 