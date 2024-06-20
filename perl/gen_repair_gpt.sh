MODEL_NAME="mistral-7b"
DATASET="humaneval"
python gen_repair_gpt.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \
