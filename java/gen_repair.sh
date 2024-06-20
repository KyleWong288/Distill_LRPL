RUN_NAME="e5-lr5"
MODEL_NAME="mistral-7b"
CHECKPOINT="checkpoint-200"
DATASET="mbxp"
TEST_DIR="ft_mm"
CUDA_VISIBLE_DEVICES=0 python gen_repair.py \
    --run_name $RUN_NAME \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --checkpoint_path "./finetuned_models/$MODEL_NAME/$RUN_NAME/$CHECKPOINT" \
    --num_rounds 4 \
    --test_dir $TEST_DIR \