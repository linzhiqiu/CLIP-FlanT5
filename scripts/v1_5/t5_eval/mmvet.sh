#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path zhiqiulin/clip-flant5-xxl \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/clip-flant5-xxl.jsonl \
    --temperature 0 \
    --conv-mode t5_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/clip-flant5-xxl.jsonl \
    --dst ./playground/data/eval/mm-vet/results/clip-flant5-xxl.json

