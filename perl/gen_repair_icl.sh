MODEL_NAME="mistral-7b"
DATASET="humaneval"
CUDA_VISIBLE_DEVICES=5 python gen_repair_icl.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_rounds 4 \