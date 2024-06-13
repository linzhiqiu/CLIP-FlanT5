## Stage-2 Training Data
The stage-2 pretraining dataset followed LLaVA-1.5 but we flattened all multi-turn conversation into single-turn in order to more efficiently train CLIP-FlanT5 (an encoder-decoder language model).
| Data file name | Size |
| --- | ---: |
| [llava_v1_5_mix665k_flattened_multi_turn.json](https://huggingface.co/datasets/zhiqiulin/CLIP-FlanT5/resolve/main/llava_v1_5_mix665k_flattened_multi_turn.json) | 1.6 GB |

## Stage-1 Training Data
The stage-1 pretraining dataset followed LLaVA-1.5. 

| Data | Chat File | Meta Data | Size |
| --- |  --- |  --- | ---: |
| LAION/CC/SBU BLIP-Caption Concept-balanced 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) | [metadata.json](#) | 181 MB

**Important notice**: Upon the request from the community, as ~15% images of the original CC-3M dataset are no longer accessible, we upload [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the CC-3M license. This may be taken down at any time when requested by the original CC-3M dataset owner or owners of the referenced images.