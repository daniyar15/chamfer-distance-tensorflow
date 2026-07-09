#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# CUDA >= 12.8 emits harmless '+ptx85' ptxas warnings during runtime JIT
# compilation for non-Blackwell GPUs.  Filter them from stderr.
python test.py 2> >(grep -v "+ptx85" >&2)