#!/bin/bash
max_step=400000
auto_step=1
max_epoch=3
len=100
lr=0.001
clip=0.25
dropout=0.1
batch_size=30
alpha=0.1
student_ratio=3
start_epoch=-1
T=1.5
gpu=3
prefix='3.lft_new_arch'
experiment_name=${prefix}_transfomer_wt103_lft_${max_step}_${max_epoch}_${len}_${batch_size}_${alpha}_${student_ratio}_${start_epoch}_${T}_${lr}_${clip}_${dropout}_${gpu}
echo 'Run training...'
log_folder_name=logs
if [ ! -d ${log_folder_name} ]; then
    mkdir -p ${log_folder_name}
fi
log_filename=${log_folder_name}/${experiment_name}.log
nohup python -u pytorch/train_lft.py \
    --cuda \
    --gpu ${gpu} \
    --data data/wikitext-103/ \
    --dataset wt103 \
    --n_layer 4 \
    --d_model 410 \
    --n_head 12 \
    --d_head 41 \
    --d_inner 2100 \
    --dropout ${dropout} \
    --dropatt 0.0 \
    --optim adam \
    --lr ${lr} \
    --clip ${clip} \
    --dropout ${dropout} \
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
    --auto_step ${auto_step} \
    --seed 1 \
    > ${log_filename} 2>&1 &
    # --multi_gpu \
    # --gpu0_bsz 4 \
    # --adaptive \
    