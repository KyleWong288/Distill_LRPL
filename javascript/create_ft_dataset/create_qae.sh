MODEL_NAME="mistral-7b"
DATA_PATH="./data/train.jsonl"
CUDA_VISIBLE_DEVICES=0 python create_qae.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \