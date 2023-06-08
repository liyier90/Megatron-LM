#! /bin/bash
set -o pipefail

DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
CHECKPOINT_PATH=12_layer_ckpt

NUM_NEURONCORES=32
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

export NEURON_NUM_RECENT_MODELS_TO_KEEP=3
export NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1

export NEURON_RT_STOCHASTIC_ROUNDING_SEED=0
export XLA_USE_BF16=1
export NEURON_CC_FLAGS="--model-type transformer"

TRAIN_ITERS=1000
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    TRAIN_ITERS=65
fi
eval_iters=250
save_interval=500

num_dim=1024
num_heads=16
num_layers=24

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --num-layers $num_layers \
    --hidden-size $num_dim \
    --num-attention-heads $num_heads \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --data-path $DATA_PATH \
    --vocab-file ~/examples_datasets/gpt2/gpt2-vocab.json \
    --merge-file ~/examples_datasets/gpt2/gpt2-merges.txt \
    --data-impl mmap \
    --split 100,0,0 \
    --distributed-backend xla \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1 \
    --lr-warmup-fraction .01 \
    --log-interval 1 \
    --tensorboard-log-interval 1 \
    --eval-interval $TRAIN_ITERS \
    --eval-iters $eval_iters \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --no-gradient-accumulation-fusion \
    --no-contiguous-buffers-in-local-ddp \
    --save-xser $CHECKPOINT_PATH \
    --save-interval $save_interval \
    --use-cpu-initialization \
    --tensorboard-dir ./tb_gpt3_24layer_bf16 \
    |& tee run_log_gpt3_24layer_bf16 &
wait %1

ret_val=$?
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

exit $ret_val

