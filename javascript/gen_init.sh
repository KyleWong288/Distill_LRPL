MODEL_NAME="mistral-7b"
DATASET="mbxp"
CUDA_VISIBLE_DEVICES=2 python gen_init.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --data_path "./data/$DATASET/test.jsonl" \