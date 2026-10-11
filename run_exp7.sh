#!/bin/bash
set -e
source /opt/pytorch/bin/activate
cd /home/ubuntu/CMCNet-Bridge-Crack-segmentation

echo "=== STARTING CMC-NET EXPERIMENT 7 (BS-GDA FUSION) TRAINING ===" | tee -a training.log
echo "Start Time: $(date)" | tee -a training.log

/opt/pytorch/bin/python -u train.py \
    --epochs 100 \
    --imgsz 512 \
    --batchsz 2 \
    --accum_steps 2 \
    --fusion bsgda 2>&1 | tee -a training.log

echo "=== RUNNING EVALUATION ON TEST SET ===" | tee -a training.log
/opt/pytorch/bin/python -u test.py \
    --checkpoint CMCNet_BSGDA_best_f1.pth 2>&1 | tee -a training.log

echo "Training and evaluation finished at: $(date)" | tee -a training.log
echo "Shutting down EC2 instance to prevent idle billing..." | tee -a training.log
sudo shutdown -h now
