#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path zhiqiulin/clip-flant5-xxl \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/clip-flant5-xxl.jsonl \
    --temperature 0 \
    --conv-mode t5_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment clip-flant5-xxl

cd eval_tool

python calculation.py --results_dir answers/clip-flant5-xxl
