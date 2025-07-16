import streamlit as st
import os
import shutil
import zipfile
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import face_recognition
import warnings
import gc
from collections import defaultdict
import cv2
import threading
from queue import Queue, Empty
import atexit

# Set LOKY_MAX_CPU_COUNT to avoid issues on Windows with wmic
if os.name == 'nt': # Check if the OS is Windows
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)

# Suppress the specific UserWarning from face_recognition_models
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

# Make sure all custom modules are in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facesorter.config import OUTPUT_DIR, TEMP_UPLOAD_DIR, TEMP_CROP_DIR, BATCH_SIZE
from facesorter.face_detector import FaceDetector
from facesorter.face_clusterer import FaceClusterer
from facesorter.file_organizer import FileOrganizer
from facesorter.media_processor import MediaProcessor
from facesorter.worker import init_worker, _process_image_worker

# --- Background File Operations ---
def file_op_worker(q):
    """Worker to process file operations from a queue in the background."""
    while True:
        try:
            op, args = q.get(block=True)
            
            if op == 'remove_file':
                file_path, = args
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            elif op == 'rename_file':
                old_path, new_path = args
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)

            elif op == 'move_files_and_remove_dir':
                src_dir, dest_dir = args
                if os.path.isdir(src_dir) and os.path.isdir(dest_dir):
                    for filename in os.listdir(src_dir):
                        shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))
                    os.rmdir(src_dir)

            elif op == 'rename_dir':
                old_path, new_path = args
                if os.path.isdir(old_path):
                    os.rename(old_path, new_path)
            
            elif op == 'remove_dir':
                dir_path, = args
                cleanup_directory(dir_path)

            q.task_done()
        except Empty:
            continue # Should not happen with block=True, but good practice
        except Exception as e:
            # Log error to console for debugging
            print(f"Background file operation error: {e}")

def start_file_worker():
    """Starts the file worker thread if it's not already running."""
    if 'file_worker_thread_running' not in st.session_state or not st.session_state.file_worker_thread_running:
        q = st.session_state.file_op_queue
        thread = threading.Thread(target=file_op_worker, args=(q,), daemon=True)
        thread.start()
        st.session_state.file_worker_thread_running = True

# --- Constants and Configuration ---
# BATCH_SIZE is now imported from config

# --- Helper Functions ---

def _process_encoding_worker(args):
    """
    Helper for multiprocessing. Encodes and crops faces for a single image,
    using pre-detected face locations.
    """
    image_path, face_locations = args
    try:
        image = face_recognition.load_image_file(image_path)
        if face_locations:
            encodings = face_recognition.face_encodings(image, face_locations)
            crops = FaceDetector.crop_faces(image, face_locations)
            if encodings:
                return (image_path, encodings, crops)
    except Exception as e:
        print(f"Encoding worker failed on {os.path.basename(image_path)}: {e}")
    return None

def cleanup_directory(dir_path):
    """Removes a directory and all its contents if it exists."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def create_zip_archive(src_dir, zip_filepath):
    """Creates a zip archive from a source directory."""
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_path = os.path.relpath(file_path, src_dir)
                zipf.write(file_path, archive_path)
    return zip_filepath

@st.cache_data
def load_and_verify_image(file_path):
    """
    Loads and verifies an image file. Caches the result.
    Returns True if the image is valid, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image is not corrupt
        return True
    except (IOError, SyntaxError, UnidentifiedImageError):
        return False

def find_unique_name(directory, desired_name):
    """
    Finds a unique directory name to avoid collisions.
    If 'desired_name' exists, it appends '_1', '_2', etc.
    """
    counter = 1
    new_name = desired_name
    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{desired_name}_{counter}"
        counter += 1
    return new_name

# --- Main Application Logic ---
def run_processing_pipeline(uploaded_files, eps_value, min_face_area, face_detector_model, max_workers):
    """
    Executes the full face detection, clustering, and sorting pipeline.
    """
    try:
        # --- INITIALIZATION ---
        cleanup_directory(TEMP_UPLOAD_DIR)
        cleanup_directory(TEMP_CROP_DIR)
        os.makedirs(TEMP_UPLOAD_DIR)
        os.makedirs(TEMP_CROP_DIR)

        # Initialize debug info holder in session state
        st.session_state.debug_info = {}
    
        face_clusterer = FaceClusterer(eps=eps_value, min_samples=1)
        file_organizer = FileOrganizer(output_dir=OUTPUT_DIR)
        
        all_encodings = []
        file_face_map = {} # Maps original file paths to their encodings
        face_data = [] # List of tuples: (encoding, crop_image)

        progress_bar = st.progress(0, text="Saving uploaded files...")

        # 1. Save all uploaded files to a temporary directory first.
        temp_file_paths = []
        video_source_map = {} # Maps extracted frame paths back to their original video path

        for uploaded_file in uploaded_files:
            temp_file_path = Path(TEMP_UPLOAD_DIR) / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if temp_file_path.suffix.lower() in ['.mp4']:
                st.info(f"Extracting frames from {temp_file_path.name}...")
                video_frame_output_dir = Path(TEMP_UPLOAD_DIR) / "video_frames"
                extracted_frames = MediaProcessor.extract_frames_from_video(temp_file_path, video_frame_output_dir)
                temp_file_paths.extend(extracted_frames)
                # For each extracted frame, store a mapping to the original video file
                for frame in extracted_frames:
                    video_source_map[frame] = temp_file_path
            else:
                temp_file_paths.append(temp_file_path)

        # 2. Process images using the selected model in parallel, with real-time progress.
        successful_results = []
        total_files = len(temp_file_paths)
        processed_files_count = 0

        st.info(f"Found {total_files} files to process. Starting face detection...")

        # Main progress bar
        progress_bar = st.progress(0, text="Initializing...")

        with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_worker,
                initargs=(face_detector_model,)
            ) as executor:
        
            futures = []
            for path in temp_file_paths:
                original_media_path = video_source_map.get(path, path)
                future = executor.submit(_process_image_worker, (path, min_face_area, original_media_path))
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                processed_files_count += 1
                
                if result:
                    successful_results.append(result)

                # Update progress bar with granular information
                progress_percentage = processed_files_count / total_files
                progress_text = f"Processing file {processed_files_count}/{total_files}: {os.path.basename(result[4].name if result else '...')}"
                progress_bar.progress(progress_percentage, text=progress_text)

        st.info("Gathering results...")
        cluster_to_original_files = defaultdict(set)
        
        # Rebuild lists from only the successful results to ensure consistency
        all_encodings = []
        face_data = []

        for i, result in enumerate(successful_results):
            original_media_path, encodings, crop_paths, debug_info, processed_path = result
            
            # This is the key fix: use extend, not append, to keep the list flat
            all_encodings.extend([(encoding, original_media_path) for encoding in encodings])
            
            if processed_path and debug_info:
                 st.session_state.debug_info[os.path.basename(str(processed_path))] = debug_info
            
            # This logic also needs to be correct
            face_data.extend([(encoding, crop_path) for encoding, crop_path in zip(encodings, crop_paths)])

            progress_bar.progress((i + 1) / len(successful_results), text=f"Gathering results {i+1}/{len(successful_results)}...")

        if not all_encodings:
            st.warning("No faces were found in any of the uploaded images.")
            cleanup_directory(TEMP_UPLOAD_DIR)
            return None, 0, [], []
        
        progress_bar.progress(1.0, text="Clustering faces...")
        # We need to unpack just the encodings for the clustering algorithm
        just_encodings = [item[0] for item in all_encodings]
        cluster_labels, num_clusters = face_clusterer.cluster_faces(just_encodings)
        
        # Now, map the original media files to their clusters
        for i, label in enumerate(cluster_labels):
            original_media_path = all_encodings[i][1]
            cluster_to_original_files[label].add(original_media_path)

        # Create a map of representative face crops for each cluster
        cluster_representatives = {}
        if num_clusters > 0:
            crop_dir = os.path.join(OUTPUT_DIR, "face_previews")
            os.makedirs(crop_dir, exist_ok=True)
            for i, label in enumerate(cluster_labels):
                if label not in cluster_representatives:
                    # Use the face_data list, which is correctly indexed
                    source_crop_path = face_data[i][1]
                    dest_crop_path = os.path.join(crop_dir, f"rep_{label}.jpg")
                    shutil.copy(source_crop_path, dest_crop_path)
                    # The UI expects the full path for the representative face
                    cluster_representatives[label] = dest_crop_path

        # This part is to restore compatibility with the old UI logic
        merge_candidates = []
        cluster_to_dest_files = file_organizer.organize_files_into_folders(cluster_to_original_files)
        
        people = {}
        if cluster_to_dest_files:
            for cluster_id, files in sorted(cluster_to_dest_files.items()):
                people[cluster_id] = {
                    "name": f"Person_{cluster_id + 1}",
                    "files": files,
                    "representative_face": cluster_representatives.get(cluster_id)
                }
        
        num_clusters = len(cluster_representatives)

        cleanup_directory(TEMP_UPLOAD_DIR)
        cleanup_directory(TEMP_CROP_DIR)

        return people, num_clusters, merge_candidates, temp_file_paths
    
    except OperationCanceledError:
        st.info("Operation cancelled by user.")
        return None, 0, [], []
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        # Optionally, log the full traceback for debugging
        # logger.exception("Unhandled error in run_processing_pipeline")
        return None, 0, [], []


def run_diagnostic_tool():
    """
    Streamlit UI for the Face Detection Diagnostic Tool.
    """
    st.header("Face Detection Diagnostic Tool")
    st.info(
        "Upload an image to see how the face detection model identifies faces. "
        "You can adjust the settings to see how they affect detection. "
        "Faces with a green box are considered 'valid' based on the current settings. "
        "Faces with a red box are detected but filtered out."
    )

    # --- UI for settings ---
    st.sidebar.header("Diagnostic Settings")
    
    # Model selection
    face_detector_model = st.sidebar.selectbox(
        "Face Detection Model",
        ("hog", "cnn"),
        index=0, # Default to 'hog'
        help="'hog' is faster but less accurate. 'cnn' is a more accurate deep learning model but much slower on CPU.",
        key="diagnostic_model" # Unique key
    )

    # Min face area slider
    min_face_area = st.sidebar.number_input(
        "Minimum Face Area (pixels)",
        min_value=0,
        max_value=3000000,
        value=300000,
        step=1000,
        help="Filters out detected faces smaller than this area (width * height).",
        key="diagnostic_min_area" # Unique key
    )
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        key="diagnostic_uploader" # Unique key
    )

    if uploaded_file is not None:
        # To use the face detector, we need to save the file to a temporary path
        temp_dir = Path(TEMP_UPLOAD_DIR)
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption=f"Original Image: {uploaded_file.name}", use_container_width=True)

        with st.spinner("Detecting faces..."):
            try:
                # --- Run Detection ---
                detector = FaceDetector(model=face_detector_model)
                # We pass 0 for min_face_area to get ALL faces, then filter visually.
                image_data, _, _, debug_info = detector.detect_faces(
                    temp_file_path,
                    min_face_area=0 
                )

                if image_data is None:
                    st.error("Could not read the uploaded image file. It might be corrupt.")
                elif not debug_info:
                    st.warning("No faces were detected in the image.")
                else:
                    st.success(f"Detected {len(debug_info)} face(s).")
                    
                    # --- Draw Boxes and Display ---
                    tagged_image = FaceDetector.draw_diagnostic_tags(
                        image_data,
                        debug_info,
                        min_face_area
                    )
                    
                    st.image(tagged_image, caption="Detection Results", use_container_width=True)
                    
                    # Display summary of faces
                    valid_faces = [info for info in debug_info if info[1] >= min_face_area]
                    st.metric(label="Valid Faces Found", value=f"{len(valid_faces)} / {len(debug_info)}")

                    st.subheader("Detected Face Details")
                    if not debug_info:
                        st.info("No faces to detail.")
                    else:
                        for i, (loc, area) in enumerate(debug_info):
                            is_valid = area >= min_face_area
                            status = "✅ Valid" if is_valid else "❌ Below Threshold"
                            st.markdown(f"**Face {i+1}**: Area = `{int(area)}` pixels ({status})")

            except Exception as e:
                st.error(f"An error occurred during face detection: {e}")
            finally:
                # Clean up the temp file
                if temp_file_path.exists():
                    os.remove(temp_file_path)


class OperationCanceledError(Exception):
    """Exception raised when the user cancels the operation."""
    pass


# --- Streamlit UI ---
def main():
    """Streamlit web interface for the FaceSorter application."""
    st.set_page_config(page_title="FaceSorter", layout="wide")

    # Register a cleanup function to be called upon script exit.
    # This will attempt to remove the sorted output directory.
    # Note: This may not run if the app is terminated forcefully.
    # atexit.register(cleanup_directory, OUTPUT_DIR) # Disabling this as it causes premature deletion

    # --- Initialize Session State & Background Worker ---
    if 'file_op_queue' not in st.session_state:
        st.session_state.file_op_queue = Queue()
    if 'file_worker_thread_running' not in st.session_state:
        st.session_state.file_worker_thread_running = False
    
    start_file_worker() # Now it's safe to call this.

    st.sidebar.title("⚙️ Settings")
    app_mode = st.sidebar.radio(
        "Choose the app mode",
        ("Face Sorter", "Diagnostic Tool")
    )

    if app_mode == "Face Sorter":
        st.title("📷 FaceSorter")
        st.write("Upload a batch of photos, and this tool will automatically sort them into folders based on the people identified in them.")

        # --- Settings Sidebar ---
        model_choice = st.sidebar.selectbox(
            "Face Detection Model",
            ("hog", "cnn"),
            index=0, # Default to hog
            help="Choose the model for detecting faces. 'hog' is faster and works well for clear, front-facing photos. 'cnn' is a more powerful deep learning model that is better at detecting faces at various angles (like profiles) but is much slower (a GPU is recommended)."
        )

        # Set a smart default for eps based on the chosen model
        default_eps = 0.4 if model_choice == 'cnn' else 0.6
        eps_help_text = (
            "Controls how similar faces must be to be grouped. Lower is stricter."
            " • If different people are in the same group, **decrease** this value."
            " • If the same person is in multiple groups, **increase** this value."
            " • Recommended start for 'cnn' is ~0.4. Recommended for 'hog' is ~0.6."
        )

        eps_value = st.sidebar.slider(
            "Clustering Sensitivity (eps)",
            min_value=0.1,
            max_value=1.0,
            value=default_eps, # Smart default
            step=0.01,
            help=eps_help_text
        )

        max_workers = st.sidebar.slider(
            "Parallel Workers",
            min_value=1,
            max_value=os.cpu_count() or 1,
            value=max(1, (os.cpu_count() or 1) // 2),  # Default to half the CPU cores
            step=1,
            help="Number of parallel processes to use for face detection. More workers can be faster but will use more RAM. Reduce this if you experience crashes with many files."
        )

        show_debugger = st.sidebar.checkbox("Show Face Size Debugger")

        min_face_area = st.sidebar.number_input(
            "Minimum Face Area (pixels)",
            min_value=0,
            max_value=3000000,
            value=300000,
            step=1000,
            help="Sets the minimum size for a face to be detected. Faces with a pixel area (width * height) smaller than this value will be ignored. Higher values will filter out more (smaller) faces."
        )

        # --- File Uploader ---
        uploaded_files = st.file_uploader(
            "Choose images or videos to sort",
            type=['jpg', 'jpeg', 'png', 'mp4'],
            accept_multiple_files=True
        )

        if uploaded_files:
            # --- File Display Options ---
            st.sidebar.write("---")
            st.sidebar.subheader("File Display Options")
            display_count_option = st.sidebar.radio(
                "Show per page",
                options=['1', '5', '10', '25', 'All'],
                index=2, # Default to 10
                horizontal=True
            )
            
            # Initialize session state for pagination and display options
            if 'page_number' not in st.session_state:
                st.session_state.page_number = 0
            if 'display_count_option' not in st.session_state:
                st.session_state.display_count_option = display_count_option

            # Reset page number if the display count changes, to prevent an invalid state
            if st.session_state.display_count_option != display_count_option:
                st.session_state.page_number = 0
                st.session_state.display_count_option = display_count_option

            display_count = len(uploaded_files) if display_count_option == 'All' else int(display_count_option)

            # --- Paginated File Display ---
            start_index = st.session_state.page_number * display_count
            end_index = start_index + display_count
            files_to_display = uploaded_files[start_index:end_index]

            st.write(f"Showing {start_index + 1}-{min(end_index, len(uploaded_files))} of {len(uploaded_files)} files.")
            
            # Display file previews in columns
            num_columns = min(display_count, 5) # Use at most 5 columns, or less if fewer are shown
            cols = st.columns(num_columns)
            for i, file in enumerate(files_to_display):
                with cols[i % num_columns]:
                    if file.type.startswith("image/"):
                        st.image(file, width=100)
                    else:
                        st.video(file) # Use st.video for video files
                    st.caption(file.name)

            # --- Pagination Controls ---
            col1, col2, col3 = st.columns([1, 1, 8])
            if st.session_state.page_number > 0:
                if col1.button("⬅️ Previous"):
                    st.session_state.page_number -= 1
                    st.rerun()

            if end_index < len(uploaded_files):
                if col2.button("Next ➡️"):
                    st.session_state.page_number += 1
                    st.rerun()

            st.write("---")
            if st.button("Sort Photos", type="primary", use_container_width=True):
                # Clean up previous results only when a new sort is initiated
                cleanup_directory(OUTPUT_DIR)
                with st.spinner("Analyzing and sorting your photos..."):
                    people, num_clusters, merge_candidates, temp_file_paths = run_processing_pipeline(
                        uploaded_files, eps_value, min_face_area, model_choice, max_workers
                    )

                    # Store results in session state to persist across reruns
                    if people is not None:
                        st.session_state.people = people
                        st.session_state.num_clusters = num_clusters
                        st.session_state.merge_candidates = merge_candidates
                        st.session_state.zip_archive = None
                        st.session_state.temp_paths = temp_file_paths
            
        # --- Face Size Debugger ---
        if show_debugger and 'debug_info' in st.session_state and st.session_state.debug_info:
            st.write("---")
            st.header("🔍 Face Size Debugger")
            st.info("This shows every face found in your images *before* the sensitivity filter is applied. Use this to find the size of unwanted faces and set the slider accordingly.")
            
            # Sort items by filename for consistent display
            sorted_debug_items = sorted(st.session_state.debug_info.items())

            for filename, debug_data in sorted_debug_items:
                with st.container(border=True):
                    st.subheader(filename)
                    
                    face_locations_with_areas = debug_data.get("face_locations")
                    if not face_locations_with_areas:
                        st.write("No faces detected in this image.")
                        continue
                    
                    # Find the original path to the temp file to display the image
                    original_path = next((path for path in st.session_state.get('temp_paths', []) if os.path.basename(str(path)) == filename), None)
                    
                    if original_path and os.path.exists(original_path):
                        try:
                            # Load the image and draw the tags
                            image_data = face_recognition.load_image_file(original_path)
                            tagged_image = FaceDetector.draw_face_tags(image_data, face_locations_with_areas)
                            st.image(tagged_image, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display debug image for {filename}: {e}")

                    areas = [f"Face #{i+1}: {int(area)} pixels" for i, (loc, area) in enumerate(face_locations_with_areas)]
                    st.write("Detected face sizes (width * height):")
                    st.code("\n".join(areas))

        # --- Manual Review Section ---
        if 'merge_candidates' in st.session_state and st.session_state.merge_candidates:
            st.write("---")
            st.header("Manual Review")
            st.info("The algorithm found groups that are very similar. Review and merge them if they are the same person.")
            
            with st.form(key="merge_form"):
                # --- Display Individual Merge Candidates ---
                for i, (id1, id2) in enumerate(st.session_state.merge_candidates):
                    # Make sure both people still exist before offering to merge
                    if id1 not in st.session_state.people or id2 not in st.session_state.people:
                        continue

                    person1 = st.session_state.people[id1]
                    person2 = st.session_state.people[id2]
                    
                    st.write(f"Merge **{person2['name']}** into **{person1['name']}**?")
                    
                    col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
                    with col1:
                        if person1.get("representative_face") and os.path.exists(person1["representative_face"]):
                            st.image(person1["representative_face"], use_container_width=True)
                    with col2:
                        if person2.get("representative_face") and os.path.exists(person2["representative_face"]):
                            st.image(person2["representative_face"], use_container_width=True)

                    with col3:
                        st.radio(
                            "Action",
                            options=["Skip", "Merge", "Reject"],
                            key=f"merge_decision_{i}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                    st.write("---") # Add a separator

                # --- Form Submission Button ---
                submitted = st.form_submit_button(
                    "Apply Merge Decisions", 
                    type="primary", 
                    use_container_width=True
                )

                if submitted:
                    processed_candidates = []
                    merged_away_ids = set()
                    any_action_taken = False

                    # Iterate through all candidates to process the decisions
                    for i, (id1, id2) in enumerate(st.session_state.merge_candidates):
                        decision_key = f"merge_decision_{i}"
                        decision = st.session_state.get(decision_key, "Skip")

                        # If a person in this pair was already merged into another, skip this pair
                        if id1 in merged_away_ids or id2 in merged_away_ids:
                            continue

                        if decision == "Merge":
                            any_action_taken = True
                            person1 = st.session_state.people[id1]
                            person2 = st.session_state.people[id2]
                            
                            # The new logic is to move all files from person2's dir to person1's
                            # The file paths in the session state are inside the person's folder, so we can get the dir from the first file.
                            if person1['files'] and person2['files']:
                                person1_dir = os.path.dirname(list(person1['files'])[0])
                                person2_dir = os.path.dirname(list(person2['files'])[0])
                                
                                # Perform synchronously to avoid race conditions on UI rerun
                                if os.path.isdir(person2_dir) and os.path.isdir(person1_dir):
                                    for filename in os.listdir(person2_dir):
                                        shutil.move(os.path.join(person2_dir, filename), os.path.join(person1_dir, filename))
                                    shutil.rmtree(person2_dir)
                                
                                # Update the state immediately for UI responsiveness
                                # Calculate the new paths for the moved files
                                moved_files_new_paths = {os.path.join(person1_dir, os.path.basename(f)) for f in person2['files']}
                                st.session_state.people[id1]['files'].update(moved_files_new_paths)
                                
                                merged_away_ids.add(id2)
                            
                            processed_candidates.append((id1, id2))
                        
                        elif decision == "Reject":
                            any_action_taken = True
                            processed_candidates.append((id1, id2))

                    # After processing all decisions, clean up the state
                    if any_action_taken:
                        # Remove the people who were merged away
                        for person_id in merged_away_ids:
                            if person_id in st.session_state.people:
                                del st.session_state.people[person_id]
                        
                        # Remove the candidate pairs that were actioned
                        st.session_state.merge_candidates = [
                            cand for cand in st.session_state.merge_candidates if cand not in processed_candidates
                        ]
                        
                        st.session_state.zip_archive = None # Invalidate zip
                        st.rerun()

        # --- Display Results ---
        if 'people' in st.session_state and st.session_state.people:
            st.write("---")
            st.header("Sorted Results")
            
            # --- Download Button ---
            st.info("Actions like merging or renaming will require you to generate a new ZIP file.")
            zip_path = "sorted_photos.zip"
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Download Link", use_container_width=True, key="generate_zip"):
                    with st.spinner("Creating download archive... this may take a moment."):
                        create_zip_archive(OUTPUT_DIR, zip_path)
                        st.session_state.zip_archive = zip_path
                        st.rerun()

            with col2:
                zip_ready = st.session_state.get('zip_archive') and os.path.exists(st.session_state.get('zip_archive'))
                if zip_ready:
                    with open(st.session_state.zip_archive, "rb") as fp:
                        st.download_button(
                            label="Download All as ZIP",
                            data=fp,
                            file_name="sorted_photos.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                else:
                    st.download_button(
                        label="Download All as ZIP",
                        data=b"",
                        file_name="sorted_photos.zip",
                        mime="application/zip",
                        disabled=True,
                        use_container_width=True,
                        help="Click 'Generate Download Link' first to create the ZIP file."
                    )
            
            # --- Form for Bulk Operations ---
            with st.form(key="results_form"):
                st.info("Edit names, or mark groups for deletion. When finished, click 'Apply All Changes' at the bottom.")
                st.write("---")

                # --- Display Folders and Images ---
                for person_id, person_data in st.session_state.people.items():
                    with st.container(border=True):
                        old_person_name = person_data['name']

                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            if person_data.get("representative_face") and os.path.exists(person_data["representative_face"]):
                                st.image(person_data["representative_face"])

                        with col2:
                            # --- Renaming Input & Remove Checkbox ---
                            rename_col, delete_col = st.columns([0.8, 0.2])
                            with rename_col:
                                st.text_input(
                                    "Group Name", 
                                    value=old_person_name, 
                                    key=f"new_name_{person_id}",
                                    label_visibility="collapsed"
                                )
                            with delete_col:
                                st.checkbox("Delete", key=f"delete_{person_id}", help="Mark this group for deletion.")

                            # --- Displaying Files (Read-Only in Form) ---
                            with st.expander(f"Show {len(person_data['files'])} files"):
                                num_cols = 8
                                cols = st.columns(num_cols)
                                
                                file_list = sorted(list(person_data['files']))

                                for i, file_path_str in enumerate(file_list):
                                    col = cols[i % num_cols]
                                    file_path = Path(file_path_str)

                                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                        try:
                                            col.image(str(file_path), use_container_width=True)
                                        except Exception:
                                            col.warning(f"Invalid image: {file_path.name}")
                                    
                                    elif file_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
                                        col.video(str(file_path))
                                    else:
                                        col.warning(f"Unsupported: {file_path.name}")
                
                # --- Form Submission Logic ---
                submitted = st.form_submit_button(
                    "Apply All Changes", 
                    type="primary", 
                    use_container_width=True
                )

                if submitted:
                    any_action_taken = False
                    ids_to_delete = set()
                    renames_to_perform = []

                    # First, determine all renames and deletions from the form's state
                    for person_id, person_data in list(st.session_state.people.items()):
                        if st.session_state.get(f"delete_{person_id}"):
                            ids_to_delete.add(person_id)
                            any_action_taken = True
                        
                        old_person_name = person_data['name']
                        new_name_key = f"new_name_{person_id}"
                        new_name = st.session_state[new_name_key]
                        sanitized_new_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                        
                        if sanitized_new_name and sanitized_new_name != old_person_name:
                            renames_to_perform.append((person_id, old_person_name, sanitized_new_name))
                            any_action_taken = True
                    
                    # Perform renames first
                    for person_id, old_name, new_name in renames_to_perform:
                        if person_id in st.session_state.people: # Check if person exists
                            person_data = st.session_state.people[person_id]
                            
                            old_dir = os.path.join(OUTPUT_DIR, old_name)
                            
                            # Find a unique name for the new directory
                            unique_new_name = find_unique_name(OUTPUT_DIR, new_name)
                            new_dir = os.path.join(OUTPUT_DIR, unique_new_name)

                            # Perform rename synchronously
                            if os.path.isdir(old_dir):
                                os.rename(old_dir, new_dir)
                            
                            # Update all file paths to reflect the new directory
                            person_data['files'] = {os.path.join(new_dir, os.path.basename(f)) for f in person_data['files']}
                            
                            # Update the name in the session state to the (potentially modified) unique name
                            person_data['name'] = unique_new_name


                    # Then, perform deletions
                    for person_id in ids_to_delete:
                        # Check if the person still exists before trying to delete
                        if person_id in st.session_state.people:
                            person_data = st.session_state.people[person_id]
                            
                            # Get the directory from one of the file paths
                            if person_data['files']:
                                person_dir = os.path.dirname(list(person_data['files'])[0])
                                st.session_state.file_op_queue.put(('remove_dir', (person_dir,)))

                            del st.session_state.people[person_id]

                    # Rerun the app once if any action was taken
                    if any_action_taken:
                        st.session_state.zip_archive = None
                        st.rerun()

    elif app_mode == "Diagnostic Tool":
        st.title("Diagnostic Tool")
        run_diagnostic_tool()


if __name__ == "__main__":
    main()