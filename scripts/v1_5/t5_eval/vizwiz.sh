#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path zhiqiulin/clip-flant5-xxl \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/clip-flant5-xxl.jsonl \
    --temperature 0 \
    --conv-mode t5_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/clip-flant5-xxl.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/clip-flant5-xxl.json
