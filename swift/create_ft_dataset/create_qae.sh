MODEL_NAME="mistral-7b"
DATA_PATH="./data/train.jsonl"
TEST_DIR="test_m"
CUDA_VISIBLE_DEVICES=6 python create_qae.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --test_dir $TEST_DIR \