import os
from pathlib import Path
from typing import List

def list_files(root_folder: str) -> List[str]:
    """
    Recursively list every file under *root_folder*, returning paths
    that all start with the given root folder.

    Parameters
    ----------
    root_folder : str
        The directory to walk.

    Returns
    -------
    List[str]
        File paths such as "root_folder/subdir/file.ext".
    """
    root = Path(root_folder).resolve()

    # .rglob("*") traverses depth-first and yields Path objects
    # We convert to strings with forward slashes so the output is
    # consistent across operating systems.
    return [
        str(p.as_posix())
        for p in root.rglob("*")
        if p.is_file()
    ]

root_fld = "/data/Synthetic/Idifface"
files = list_files(root_fld)


with open(os.path.join(root_fld, 'image_path_list.txt'), 'w') as f:
    for fpath in files:
        #fpath = "/".join(fpath.split("/")[2:])
        f.write(fpath+"\n")
        print(fpath)
