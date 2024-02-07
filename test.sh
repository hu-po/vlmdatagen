#!/bin/bash
LLM="gpt"
DATA_DIR="/home/oop/dev/data/testgencap"
python gensdxl.py \
  --seed 0 \
  --data_dir "$DATA_DIR" \
  --dataset_size 32 \
  --llm "$LLM" \
  --num_prompts 4
python3 capllava.py \
  --data_dir "$DATA_DIR" \
  --seed 0 \
  --llm "$LLM" \
  --num_prompts 4