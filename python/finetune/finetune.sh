export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="e8-lr1"
MODEL_NAME="cl-7b-instruct"
DATA_PATH="../data"

CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 1600 \
    --num_epochs 8 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --warmup_steps 10 \
    --log_interval 10 \
    --scheduler cosine \
    --deepspeed ds_config.json \
