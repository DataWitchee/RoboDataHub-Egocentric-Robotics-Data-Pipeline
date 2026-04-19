import os
import zipfile
from pathlib import Path

def create_zip_archive(source_dir: Path, output_path: str):
    """
    Zips the contents of source_dir and saves it to output_path.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
        
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Create arcname so zip has relative paths inside
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
