MODEL_NAME="mistral-7b"
DATASET="humaneval"
CUDA_VISIBLE_DEVICES=4 python gen_repair_base.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \