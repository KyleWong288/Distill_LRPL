MODEL_NAME="mistral-7b"
DATASET="mbxp"
CUDA_VISIBLE_DEVICES=0 python gen_repair_base.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \