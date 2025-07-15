import os
import shutil
from collections import defaultdict
from pathlib import Path

class FileOrganizer:
    """
    Organizes files based on face clustering results.
    """

    def __init__(self, output_dir="sorted_output"):
        """
        Initializes the FileOrganizer.

        Args:
            output_dir (str): The root directory where sorted files will be stored.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def organize_files_into_folders(self, cluster_to_files):
        """
        Organizes media files into person-specific folders, keeping original filenames.

        Args:
            cluster_to_files (dict): A dictionary mapping each cluster ID to the set of original
                                     media file paths that contain a face from that cluster.

        Returns:
            dict: A dictionary mapping each cluster ID to the set of DESTINATION file paths.
        """
        cluster_to_dest_files = defaultdict(set)
        for cluster_id, files in cluster_to_files.items():
            person_name = f"Person_{cluster_id + 1}"
            person_dir = os.path.join(self.output_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Copy each file into the person-specific directory
            for src_path in files:
                original_file_name = os.path.basename(str(src_path))
                dest_path = os.path.join(person_dir, original_file_name)

                # Avoid re-copying if already there
                if not os.path.exists(dest_path):
                    shutil.copy(str(src_path), dest_path)
                cluster_to_dest_files[cluster_id].add(dest_path)

        return dict(cluster_to_dest_files)

    def get_all_sorted_files(self):
        """
        Retrieves a list of all file paths in the sorted output directory.
        """
        all_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files 