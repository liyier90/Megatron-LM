#!/bin/bash

checkpoint_path=/home/ubuntu/Megatron-LM/12_layer_ckpt/iter_0001000
output_path=/home/ubuntu/Megatron-LM/12_layer_ckpt/trfs_checkpoint


transformers_dir="$(dirname $(dirname $(which python)))/lib/python3.8/site-packages/transformers"

# python ${transformers_dir}/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py \
PYTHONPATH=/home/ubuntu/Megatron-LM \
        python checkpoint_reshaping_and_interoperability.py \
        --convert_checkpoint_from_megatron_to_transformers \
        --load_path $checkpoint_path \
        --save_path $output_path \
        --tokenizer_name "gpt2" \
        --print-checkpoint-structure \
