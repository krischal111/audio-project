# Dataset is the common voice corpus

# Want to separate audio to parts

from pathlib import Path
import os

def get_all_files_from(folder):
    files = []
    i = 0
    for (path, dirlist, filelist) in os.walk(folder):
        path = Path(path)
        for file in filelist:
            filepath = path.joinpath(Path(file))
            files.append(filepath)
    return files