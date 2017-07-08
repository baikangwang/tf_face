"""
outline local face images and outputs manifest_uniq.txt for crop_faces.py
1. get the list of image files
2. write image path in format "{prepend path}/{image name}"
"""
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: prepare_local_faces.py <image directory>")
        exit(0)

    dir = sys.argv[1]
    with open(os.path.join(dir, "manifest_uniq_local.txt"), 'w') as manifest:
        for f in os.listdir(dir):
            line = os.path.join(dir, f)
            if os.path.isfile(line):
                manifest.write(line + "\n")
