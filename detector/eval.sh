#!/bin/bash

DEVICE=1

LABLE_TYPE=$1
OUTPUT=output_type$1
#MODEL_PATH=./output_type$1/exp-1/model_final.pth
MODEL_PATH=./output_type$1/exp-2/model_final.pth
THRSH=0.8


CUDA_VISIBLE_DEVICES=$DEVICE python3 detection.py test --label_type $LABLE_TYPE\
	--output $OUTPUT\
	--model_path $MODEL_PATH\
	--thrsh $THRSH




# label type = 1 - 9 classes + opt unfocused
# label type = 2 - 2 classes [species] + opt unfocused
# label type = 2 - 3 classes [growth stage] + opt unfocused
