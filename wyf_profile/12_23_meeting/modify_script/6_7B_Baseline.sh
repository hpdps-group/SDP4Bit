#!/bin/bash
set -x

GPUS_PER_NODE=4
MASTER_PORT=6233
NUM_NODES=1
WORLD_SIZE=NUM_NODES*GPUS_PER_NODE


# Parallel Setting
# 这里甚至能把TP改成1 试试看
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1
MICRO_BATCH_SIZE=1
# Calculate GLOBAL_BATCH_SIZE based on Accumulation Step=1
GLOBAL_BATCH_SIZE=$(($WORLD_SIZE / ($TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE) * $MICRO_BATCH_SIZE))


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
MEGATRON_PATH="/HOME/scw6doz/run/wyf/SDP4Bit"

# Distributed training args
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# 6.7B Model
MODEL_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

# learning rate, training type and other optimizer arguments
OPTIMIZER_ARGS="
    --lr 0.00012 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.000012 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --bf16 \
    --use-distributed-optimizer \
"

# Parallel size, global batch and others, (micro batch size depeneds on gpu memory)
TRAINING_ARGS="
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 80000 \
"

# Data path, --mock-data is using fake data (to avoid IO overhead)
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --timing-log-level 2 \
    --save-interval 5002 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
"

QUANTIZE_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
"

PRETRAIN_PY="$MEGATRON_PATH/pretrain_gpt.py"

torchrun $DISTRIBUTED_ARGS $PRETRAIN_PY \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    --distributed-backend nccl \
    --exit-interval 200 \