MODEL_NAME="mistral-7b"
DATASET="humaneval"
TEST_DIR="base_h"
CUDA_VISIBLE_DEVICES=5 python gen_repair_base.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \
    --test_dir $TEST_DIR \