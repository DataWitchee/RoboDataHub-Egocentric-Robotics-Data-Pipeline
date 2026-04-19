import os
from pathlib import Path

TARGET_DIR = Path("/Users/mannatsaini/cccc/egocentric_toolkit")
OLD_PATH = '"/Users/mannatsaini/Desktop/my_robotics_data'
NEW_PATH = '"/Users/mannatsaini/Desktop/my_robotics_data'

for py_file in TARGET_DIR.glob("*.py"):
    content = py_file.read_text()
    if OLD_PATH in content:
        content = content.replace(OLD_PATH, NEW_PATH)
        py_file.write_text(content)
        print(f"Patched: {py_file.name}")
