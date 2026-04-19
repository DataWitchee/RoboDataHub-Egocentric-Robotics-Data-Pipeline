import os
import re
from pathlib import Path
from fastapi import UploadFile, HTTPException

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".avi"}

def sanitize_filename(filename: str) -> str:
    """Removes weird characters and spaces from the filename."""
    name, ext = os.path.splitext(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    return safe_name + ext.lower()

def validate_and_save_file(file: UploadFile, target_dir: Path) -> float:
    """
    Validates file extension and size.
    Saves the file to target_dir.
    Returns the file size in MB.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Validate extension
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type {ext}. Only MP4/AVI allowed.")
        
    # 2. Sanitize and define path
    safe_name = sanitize_filename(file.filename)
    file_path = target_dir / safe_name
    
    # 3. Read content to check size and save
    contents = file.file.read()
    file_size_bytes = len(contents)
    
    if file_size_bytes > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 500MB limit.")
        
    with open(file_path, "wb") as f:
        f.write(contents)
        
    file.file.seek(0)
    return file_size_bytes / (1024 * 1024)
