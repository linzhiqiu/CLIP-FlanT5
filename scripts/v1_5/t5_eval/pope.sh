#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path zhiqiulin/clip-flant5-xxl \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/clip-flant5-xxl.jsonl \
    --temperature 0 \
    --conv-mode t5_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/clip-flant5-xxl.jsonl
