#!/bin/bash
set -e
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TMPDIR=/opt/dlami/nvme/tmp
export MAMBA_KEEP_CUDA_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
cd /opt/dlami/nvme/mamba
/opt/pytorch/bin/python setup.py build_ext --inplace
