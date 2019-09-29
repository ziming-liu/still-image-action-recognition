#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 train_dist.py $1 --launcher pytorch ${@:3}