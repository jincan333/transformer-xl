#!/bin/bash
max_step=12000
max_epoch=60
len=70
batch_size=60
alpha=0.5
student_ratio=3
start_epoch=-1
T=1.5
lr=0.001
clip=0.35
dropout=0.25
gpu=0
prefix='4.student_ratio'
experiment_name=${prefix}_transfomer_ptb_lft_${max_step}_${max_epoch}_${len}_${batch_size}_${alpha}_${student_ratio}_${start_epoch}_${T}_${lr}_${clip}_${dropout}_${gpu}
echo 'Run training...'
log_filename=ptb_logs/${experiment_name}.log
nohup python -u pytorch/train_lft.py \
    --cuda \
    --gpu ${gpu} \
    --data data/ptb/ \
    --dataset ptb \
    --n_layer 4 \
    --d_model 200 \
    --n_head 8 \
    --d_head 20 \
    --d_inner 1000 \
    --dropout ${dropout} \
    --dropatt 0.0 \
    --optim adam \
    --lr ${lr} \
    --clip ${clip} \
    --warmup_step 0 \
    --max_step ${max_step} \
    --tgt_len ${len} \
    --mem_len ${len} \
    --eval_tgt_len ${len} \
    --batch_size ${batch_size} \
    --eval-interval 1000 \
    --alpha ${alpha} \
    --student_ratio ${student_ratio} \
    --T ${T} \
    --exp_name ${experiment_name} \
    --max_epoch ${max_epoch} \
    --start_epoch ${start_epoch} \
    > ${log_filename} 2>&1 &
    # --multi_gpu \
    # --gpu0_bsz 4 \
    # --adaptive \
    