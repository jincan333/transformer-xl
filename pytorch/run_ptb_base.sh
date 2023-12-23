#!/bin/bash
max_step=200000
experiment_name=transfomer_ptb_baseline_${max_step}
if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    log_filename=logs/train_${experiment_name}.log
    nohup python -u pytorch/train.py \
        --cuda \
        --gpu 2 \
        --data data/ptb/ \
        --dataset ptb \
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
        --max_step 200000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 60 \
        ${@:2} \
        > ${log_filename} 2>&1 &
        # --multi_gpu \
        # --gpu0_bsz 4 \
        
        
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    log_filename=logs/test_${experiment_name}.log
    nohup python -u pytorch/eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2} 
        > ${log_filename} 2>&1 &
        
        
else
    echo 'unknown argment 1'
fi
