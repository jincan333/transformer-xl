#!/bin/bash
max_step=300000
max_epoch=4
len=100
batch_size=20
alpha=0.1
student_ratio=4
start_epoch=-1
T=1.5
gpu=0
prefix='4.debug'
experiment_name=${prefix}_transfomer_wt103_lft_${max_step}_${max_epoch}_${len}_${batch_size}_${alpha}_${student_ratio}_${start_epoch}_${T}_${gpu}
echo 'Run training...'
log_filename=logs/${experiment_name}.log
nohup python -u pytorch/train_lft.py \
    --cuda \
    --gpu ${gpu} \
    --data data/wikitext-103/ \
    --dataset wt103 \
    --n_layer 4 \
    --d_model 200 \
    --n_head 8 \
    --d_head 20 \
    --d_inner 1000 \
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
    