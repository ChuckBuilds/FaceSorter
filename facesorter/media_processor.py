from pathlib import Path

from facesorter import HEIC_SUPPORTED


class MediaProcessor:
    """Handles the discovery of image files in a given directory."""

    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'] + \
        (['.heic', '.heif'] if HEIC_SUPPORTED else [])

    def __init__(self, source_folder):
        """
        Args:
            source_folder (str): The path to the folder containing images.
        """
        self.source_folder = Path(source_folder)
        if not self.source_folder.is_dir():
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

    def discover_media(self):
        """
        Recursively scans the source folder for supported image files.

        Returns:
            A sorted list of image file paths.
        """
        image_files = [
            p for p in self.source_folder.rglob('*')
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS
        ]
        return sorted(image_files)
