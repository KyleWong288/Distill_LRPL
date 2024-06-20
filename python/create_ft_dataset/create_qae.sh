MODEL_NAME="cl-7b-instruct"
DATA_PATH="./data/train.jsonl"
CUDA_VISIBLE_DEVICES=6 python create_qae.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \