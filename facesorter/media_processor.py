import os
from pathlib import Path
import cv2
import time

class MediaProcessor:
    """Handles the discovery of media files in a given directory."""

    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.heic']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv']

    def __init__(self, source_folder):
        """
        Initializes the MediaProcessor with a source folder.

        Args:
            source_folder (str): The path to the folder containing media files.
        """
        self.source_folder = Path(source_folder)
        if not self.source_folder.is_dir():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

    def discover_media(self):
        """
        Scans the source folder for supported image and video files.

        Returns:
            A tuple containing two lists: one for image paths and one for video paths.
        """
        image_files = []
        video_files = []
        for file_path in self.source_folder.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS:
                    image_files.append(file_path)
                elif file_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS:
                    video_files.append(file_path)
        return image_files, video_files

    @staticmethod
    def extract_frames_from_video(video_path, output_folder, frames_per_second=1):
        """
        Extracts frames from a video file and saves them as images.

        Args:
            video_path (str or Path): The path to the video file.
            output_folder (str or Path): The directory to save the extracted frames.
            frames_per_second (int): How many frames to extract per second of video.

        Returns:
            list: A list of Path objects for the extracted frame images.
        """
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        extracted_frames = []
        
        try:
            vid_cap = cv2.VideoCapture(str(video_path))
            if not vid_cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return []

            video_fps = vid_cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                print(f"Warning: Video FPS is 0 for {video_path.name}. Defaulting to 30.")
                video_fps = 30 # Default to a common FPS

            # Calculate the frame interval to achieve the desired frames_per_second
            frame_interval = int(video_fps / frames_per_second)
            if frame_interval == 0:
                frame_interval = 1 # Ensure we extract at least one frame if fps is low

            frame_count = 0
            while True:
                success, image = vid_cap.read()
                if not success:
                    break # End of video
                
                if frame_count % frame_interval == 0:
                    # Construct a unique filename
                    timestamp = int(time.time() * 1000)
                    frame_filename = f"{video_path.stem}_frame_{frame_count}_{timestamp}.jpg"
                    frame_path = output_folder / frame_filename
                    
                    # Save the frame as a high-quality JPEG
                    cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_frames.append(frame_path)
                
                frame_count += 1
        
        finally:
            if 'vid_cap' in locals() and vid_cap.isOpened():
                vid_cap.release()

        return extracted_frames 