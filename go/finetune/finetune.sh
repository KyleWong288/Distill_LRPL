export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
RUN_NAME="e5-lr5-mgo"
MODEL_NAME="mistral-7b"
DATA_PATH="../data"

CUDA_VISIBLE_DEVICES=2,3 python finetune.py \
    --run_name $RUN_NAME \
    --base_model $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_seq_length 1600 \
    --num_epochs 5 \
    --learning_rate 5e-6 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --warmup_steps 10 \
    --log_interval 10 \
    --scheduler cosine \
    --deepspeed ds_config.json \
