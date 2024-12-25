#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
#MASTER_ADDR=10.254.128.14/
MASTER_PORT=6233
NUM_NODES=4
#NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


# 数据集还是用原来的,保存在自己的路径下测试用.
CHECKPOINT_PATH="/HOME/scw6doz/run/wyf/dataset/wikitext/V1_0/checkpoints"
#<Specify path>
VOCAB_FILE="/HOME/scw6doz/run/zly/dataset/wikitext/V1_0/vocab.json"
#<Specify path to file>/gpt2-vocab.json
MERGE_FILE="/HOME/scw6doz/run/zly/dataset/wikitext/V1_0/merges.txt"
#<Specify path to file>/gpt2-merges.txt
DATA_PATH="/HOME/scw6doz/run/zly/dataset/wikitext/V1_0/meg-gpt2_text_document"
#<Specify path and file prefix>_text_document
TENSORBOARD_DIR="//HOME/scw6doz/run/wyf/SDP4Bit/wyf_profile/log"
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 200 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun ${DISTRIBUTED_ARGS[@]} /HOME/scw6doz/run/wyf/SDP4Bit/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH \
    # --use-pytorch-profiler \
    # --tensorboard-dir ${TENSORBOARD_DIR}"
