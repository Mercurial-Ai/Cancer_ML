#!/bin/sh
readarray -d '' paths < <(find . -maxdepth 3 -mindepth 3 -type d)
for path in ${paths[@]}
do
    base=$(echo "$path" | cut -d "/" -f2)
    save="mesh.stl"
    save_path="$base$save"
    echo $save_path
    dicom2mesh -i $path -t 557 -o $save_path
done