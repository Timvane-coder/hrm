#!/bin/bash

# 3D Pose Estimation using hmr2.0 (Python 3 / TF2 replacement for original hmr)
#
# Drop-in replacement for the original 3dpose_estimate.sh
# Output: one CSV per image in hmr/output/csv/  (same as before)

FRAME=0

for f in keras_Realtime_Multi-Person_Pose_Estimation/sample_images/*; do

    filename=$(basename -- "$f")
    no_ext="${filename%.*}"

    echo "Processing $no_ext (frame $FRAME)"

    python hmr/3dpose_estimate.py \
        --img_path  "$f" \
        --json_path "keras_Realtime_Multi-Person_Pose_Estimation/sample_jsons/${no_ext}.json" \
        --out_dir   "output/csv" \
        --frame     "$FRAME"

    FRAME=$((FRAME + 1))

done

echo "Done — $FRAME frames processed"
