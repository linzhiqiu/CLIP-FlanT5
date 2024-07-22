# Model Zoo
We recommend using the largest clip-flant5-xxl (11B) for optimal performance.

## CLIP-FlanT5 (Stage-2 VQA)

| Version | Size | Checkpoint | Winoground | EqBen-Mini | DrawBench | EditBench | COCO-T2I | TIFA160 | Pick-a-Pic | GenAI-Bench-Image | GenAI-Bench-Video |
|----------|----------|-----------|-----------|---|---|---|---|---|---|---|---|
| CLIP-FlanT5-XXL | 11B | [zhiqiulin/clip-flant5-xxl](https://huggingface.co/zhiqiulin/clip-flant5-xxl) | 46.0 | 47.9 | 85.3 | 77.0 | 85.0 | 71.2 | 84.0 | 64.1 | 63.2 |
| CLIP-FlanT5-XL | 3B  | [zhiqiulin/clip-flant5-xl](https://huggingface.co/zhiqiulin/clip-flant5-xl) | 34.8 | 39.3 | 82.8 | 74.5 | 80.7 | 68.8 | 84.0 | 61.8 | 60.2 |


## Projector weights (Stage-1 Captioning)

These are projector weights we have pretrained. 

NOTE: When you use our pretrained projectors for training on VQA data, it is very important to use the same base LLM (FlanT5) and vision encoder as the one we used for pretraining the projector. Otherwise, the performance will be very poor.

| Base LLM | Vision Encoder | Projection | Pretrain Data | Pretraining schedule | Download |
|----------|----------------|---------------|----------------------|----------|----------|
| FlanT5-11B | CLIP-L-336px | MLP-2x | LCS-558K | 1e | [projector](https://huggingface.co/zhiqiulin/clip-flant5-xxl-stage-1) |
| FlanT5-3B | CLIP-L-336px | MLP-2x | LCS-558K | 1e | [projector](https://huggingface.co/zhiqiulin/clip-flant5-xl-stage-1) |