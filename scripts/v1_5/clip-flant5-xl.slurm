#!/bin/bash
#SBATCH --job-name=clip-flant5-xl
#SBATCH --partition=onevision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 4-00:00:00            # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name

echo "START TIME: $(date)"

export NCCL_DEBUG=WARN
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

srun --jobid $SLURM_JOB_ID bash -c 'python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/t5_train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path google/flan-t5-xl \
    --version t5_v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_flattened_multi_turn.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/clip-flant5-xl-stage-1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/clip-flant5-xl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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