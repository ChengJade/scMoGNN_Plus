#!/bin/sh
export root_path="/home/chengyu/Pycharm/Predict_Modality/"
for i in {1..10}
do
	echo "$i"
	    python "$root_path/generate_extra_files.py" -i "$i"
    done
