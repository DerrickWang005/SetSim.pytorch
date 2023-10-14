#!/bin/bash
python main_pretrain.py \
    --aug-plus \
    --cos \
    -a resnet50 \
    -j 32 \
    -p 100 \
    --lr 0.06 \
    --batch-size 512 \
    --epochs 200 \
    --dist-url 'tcp://localhost:6681' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --seed 0 \
    --moco-t 0.2 \
    --p-weight 0.5 \
    --attention \
    --att-threshold 0.7 \
    --geometry \
    --geo-threshold 0.5 \
    --neg 0.2 \
    $DATA_DIR
