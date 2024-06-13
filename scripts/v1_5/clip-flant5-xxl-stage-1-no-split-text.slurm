#!/bin/bash
#SBATCH --job-name=clip-flant5-xxl-stage-1-no-split-text
#SBATCH --partition=onevision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 4-00:00:00            # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name

echo "START TIME: $(date)"

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

srun --jobid $SLURM_JOB_ID bash -c 'python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/t5_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path google/flan-t5-xxl \
    --version t5_plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/clip-flant5-xxl-stage-1-no-split-text \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-2 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb'


echo "END TIME: $(date)"