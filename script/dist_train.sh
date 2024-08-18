#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0,1

gpu_num=${#gpus[@]}
echo "[number of gpus]: "${gpu_num}

port=${PORT:-28650}
echo "[port of ddp]: "${port}

config=$1

python3 -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=$port \
    script/train.py $config --launcher pytorch ${@:2}
