import os
import argparse
from .config import config
from .media_processor import MediaProcessor
from .face_detector import FaceDetector

def main():
    """Main function to run the face sorter."""
    parser = argparse.ArgumentParser(description="Detects faces in media and saves the output.")
    parser.add_argument(
        "--source",
        type=str,
        default="input_media",
        help="The folder containing the media files to process."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_media",
        help="The folder where the processed media will be saved."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="The face detection model to use. 'hog' is faster; 'cnn' is more accurate but slower."
    )
    args = parser.parse_args()

    print("FaceSorter application started.")
    
    source_folder = args.source
    output_folder = args.output
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    media_processor = MediaProcessor(source_folder)
    face_detector = FaceDetector(model=args.model)
    
    print(f"Using '{args.model}' model for face detection.")
    print(f"Scanning for media in: {source_folder}")
    image_files, video_files = media_processor.discover_media()
    
    print(f"Found {len(image_files)} image files.")
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        try:
            image, face_locations = face_detector.detect_faces(img_path)

            if face_locations:
                print(f"  Found {len(face_locations)} face(s).")

                # Draw faces
                image_with_faces = FaceDetector.draw_faces(image, face_locations)

                # Save the new image
                output_path = os.path.join(output_folder, f"detected_{img_path.name}")
                image_with_faces.save(output_path)
                print(f"  Saved result to {output_path}")

            else:
                print("  No faces found.")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"Found {len(video_files)} video files.")
    for vid in video_files:
        print(f"  - {vid.name}")

if __name__ == "__main__":
    main() 