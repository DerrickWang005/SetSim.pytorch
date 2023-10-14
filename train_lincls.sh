#!/bin/bash
python main_lincls.py \
    -a resnet50 \
    -p 100 \
    --pretrained $CKPT_DIR \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    $DATA_DIR