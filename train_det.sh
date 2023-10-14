#!/bin/bash
python detection/train_net.py \
    --config-file detection/configs/pascal_voc_R_50_C4_24k_moco.yaml \
    --num-gpus 8 \
    MODEL.WEIGHTS $CKPT_DIR \
    OUTPUT_DIR $EXP_DIR