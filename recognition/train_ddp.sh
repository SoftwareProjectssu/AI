#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    main.py \
    --data_dir ../300W \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --nstack 4 \
    --num_workers 4 \
    --save_dir checkpoints \
    --log_dir runs/dvit_300w