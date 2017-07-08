#!/bin/bash

# clean up the images in broken
python tf/face_extract/clean_faces.py ./data/manifest_uniq.txt $PWD 
