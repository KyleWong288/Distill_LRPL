MODEL_NAME="mistral-7b"
DATASET="mbxp"
TEST_DIR="gpt_m"
python gen_repair_gpt.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \
    --test_dir $TEST_DIR \
