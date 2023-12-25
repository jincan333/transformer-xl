#!/bin/bash
max_step=100000
max_epoch=4
len=150
batch_size=40
alpha=0.5
student_ratio=1
start_epoch=-1
T=1
gpu=3
prefix='0.test'
experiment_name=${prefix}_transfomer_wt103_lft_${max_step}_${max_epoch}_${len}_${batch_size}_${alpha}_${student_ratio}_${start_epoch}_${T}_${gpu}
echo 'Run training...'
log_filename=logs/${experiment_name}.log
nohup python -u pytorch/train_lft.py \
    --cuda \
    --gpu ${gpu} \
    --data data/wikitext-103/ \
    --dataset wt103 \
    --adaptive \
    --n_layer 16 \
    --d_model 410 \
    --n_head 10 \
    --d_head 41 \
    --d_inner 2100 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step ${max_step} \
    --tgt_len ${len} \
    --mem_len ${len} \
    --eval_tgt_len ${len} \
    --batch_size ${batch_size} \
    --alpha ${alpha} \
    --student_ratio ${student_ratio} \
    --T ${T} \
    --exp_name ${experiment_name} \
    --max_epoch ${max_epoch} \
    --start_epoch ${start_epoch} \
    > ${log_filename} 2>&1 &
    # --multi_gpu \
    # --gpu0_bsz 4 \
    