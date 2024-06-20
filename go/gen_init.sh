MODEL_NAME="mistral-7b"
DATASET="humaneval"
CUDA_VISIBLE_DEVICES=3 python gen_init.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --data_path "./data/$DATASET/test.jsonl" \