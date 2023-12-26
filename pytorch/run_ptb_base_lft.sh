#!/bin/bash
max_step=10000
max_epoch=60
len=35
batch_size=80
alpha=0
student_ratio=0
start_epoch=-1
T=1.5
gpu=7
prefix='1.debug'
experiment_name=${prefix}_transfomer_ptb_lft_${max_step}_${max_epoch}_${len}_${batch_size}_${alpha}_${student_ratio}_${start_epoch}_${T}_${gpu}
echo 'Run training...'
log_filename=ptb_logs/${experiment_name}.log
nohup python -u pytorch/train_lft.py \
    --cuda \
    --gpu ${gpu} \
    --data data/ptb/ \
    --dataset ptb \
    --adaptive \
    --n_layer 4 \
    --d_model 410 \
    --n_head 8 \
    --d_head 41 \
    --d_inner 2100 \
    --dropout 0.45 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step ${max_step} \
    --tgt_len ${len} \
    --mem_len ${len} \
    --eval_tgt_len ${len} \
    --batch_size ${batch_size} \
    --eval-interval 200 \
    --alpha ${alpha} \
    --student_ratio ${student_ratio} \
    --T ${T} \
    --exp_name ${experiment_name} \
    --max_epoch ${max_epoch} \
    --start_epoch ${start_epoch} \
    > ${log_filename} 2>&1 &
    # --multi_gpu \
    # --gpu0_bsz 4 \
    