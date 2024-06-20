MODEL_NAME="cl-7b-instruct"
DATASET="mbxp"
TEST_DIR="test"
CUDA_VISIBLE_DEVICES=0 python gen_repair_base.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \
    --test_dir $TEST_DIR \