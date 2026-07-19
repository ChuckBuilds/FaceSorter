import os
import shutil


class FileOrganizer:
    """
    Copies media files into per-person folders. A group photo of A, B and C
    is copied into each of the three folders.
    """

    def __init__(self, output_dir="sorted_output"):
        self.output_dir = output_dir

    def export(self, people):
        """
        Copies source files into <output_dir>/<person name>/ for each group.

        Args:
            people (dict): {group_id: {"name": str, "files": set of source
                            paths}} — the reviewed groups from the UI.

        Returns:
            dict: {group_id: number of files copied} for a summary.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        copied = {}
        for group_id, person in people.items():
            person_dir = os.path.join(self.output_dir, person["name"])
            os.makedirs(person_dir, exist_ok=True)

            count = 0
            for src_path in sorted(str(p) for p in person["files"]):
                dest_path = self._unique_dest(person_dir, os.path.basename(src_path))
                if dest_path is not None:
                    shutil.copy2(src_path, dest_path)
                    count += 1
            copied[group_id] = count
        return copied

    @staticmethod
    def _unique_dest(person_dir, file_name):
        """
        Picks a destination path, suffixing _1, _2, ... when a different
        file with the same name already exists. Returns None if this exact
        file was already copied (same name and size), to keep re-exports
        idempotent.
        """
        stem, ext = os.path.splitext(file_name)
        counter = 0
        while True:
            candidate = os.path.join(
                person_dir, file_name if counter == 0 else f"{stem}_{counter}{ext}"
            )
            if not os.path.exists(candidate):
                return candidate
            counter += 1
