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

    def organize_by_cluster(self, cluster_to_files):
        """
        Copies media files into a single directory, with each file prefixed by the person's name.

        Args:
            cluster_to_files (dict): A dictionary mapping each cluster ID to the set of original
                                     media file paths that contain a face from that cluster.

        Returns:
            dict: A dictionary mapping each cluster ID to the set of DESTINATION file paths.
        """
        cluster_to_dest_files = defaultdict(set)
        for cluster_id, files in cluster_to_files.items():
            person_prefix = f"Person_{cluster_id + 1}"
            
            # Copy each file into the single output directory with a new prefixed name
            for src_path in files:
                original_file_name = os.path.basename(str(src_path))
                new_file_name = f"{person_prefix}_{original_file_name}"
                dest_path = os.path.join(self.output_dir, new_file_name)
                
                # Avoid re-copying if already there
                if not os.path.exists(dest_path):
                    shutil.copy(src_path, dest_path)
                cluster_to_dest_files[cluster_id].add(dest_path)

        return dict(cluster_to_dest_files)

    def organize_files_by_cluster(self, cluster_to_original_files, cluster_names):
        """
        Organizes files into folders named after the identified person.

        Args:
            cluster_to_original_files (dict): Maps cluster labels to original file paths.
            cluster_names (dict): Maps cluster labels to the names provided by the user.

        Returns:
            list: A list of the new paths of all sorted files.
        """
        all_sorted_files = []
        # First, clear out any existing subdirectories to prevent file duplication
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

        for cluster_id, files in cluster_to_original_files.items():
            person_name = cluster_names.get(cluster_id, f"Person_{cluster_id + 1}")
            person_dir = os.path.join(self.output_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            for file_path in files:
                dest_path = os.path.join(person_dir, Path(file_path).name)
                if not os.path.exists(dest_path):
                    shutil.copy(str(file_path), dest_path)
                all_sorted_files.append(dest_path)
        
        return all_sorted_files

    def get_all_sorted_files(self):
        """
        Retrieves a list of all file paths in the sorted output directory.
        """
        all_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files 