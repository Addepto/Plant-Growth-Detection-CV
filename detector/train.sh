#!/bin/bash

DEVICE=1

LABLE_TYPE=3
OUTPUT=output_type3

CUDA_VISIBLE_DEVICES=$DEVICE python3 detection.py train --label_type $LABLE_TYPE\
	--output $OUTPUT\
	--debug




# label type = 1 - 9 classes + opt unfocused
# label type = 2 - 2 classes [species] + opt unfocused
# label type = 2 - 3 classes [growth stage] + opt unfocused
