#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path zhiqiulin/clip-flant5-xxl \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/clip-flant5-xxl.jsonl \
    --temperature 0 \
    --conv-mode t5_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/clip-flant5-xxl.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/clip-flant5-xxl.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/clip-flant5-xxl.jsonl
