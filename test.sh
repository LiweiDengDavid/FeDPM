#!/bin/bash

# =====================================================================================
# FeDPM Training Script for Multiple Forecasting Horizons (Tout)
# =====================================================================================

# Base configuration
BASE_CUDA_ID=0
BASE_SEED=2025
NUM_ROUNDS=1
LOCAL_EPOCHS=1
AGGREGATION_STRATEGY="cos_similarity"
MEMORY_FILL_STRATEGY="client_personalized"
CLIENT_DATA_TYPES=("ETTh1" "ETTh2")
BASE_DATA_PATH="./saved_data/"
LOG_DIR="./logs/federated/"
BASE_LR=1e-5
D_MODEL=64
NHEAD=4
TIN=96

# Prediction horizons to iterate over
TOUT_VALUES=(96)

# Checkpoint and Log parameters
CHECKPOINT_PARAMS=("Tin" "Tout" "seed" "encoder_config_transformer_layers" "nlayers" "compression")
LOG_PARAMS=("Tin" "seed" "encoder_config_transformer_layers" "nlayers" "compression")

for TOUT in "${TOUT_VALUES[@]}"; do
    echo "================================================================================"
    echo "Starting Experiment: TIN=96, TOUT=$TOUT"
    echo "================================================================================"

    python federated_learning_main.py \
       --cuda-id $BASE_CUDA_ID \
       --seed $BASE_SEED \
       --num_rounds $NUM_ROUNDS \
       --local_epochs $LOCAL_EPOCHS \
       --aggregation_strategy "$AGGREGATION_STRATEGY" \
       --memory_fill_strategy "$MEMORY_FILL_STRATEGY" \
       --client_data_types "${CLIENT_DATA_TYPES[@]}" \
       --base_data_path "$BASE_DATA_PATH" \
       --log_dir "$LOG_DIR" \
       --baselr $BASE_LR \
       --d_model $D_MODEL \
       --nhead $NHEAD \
       --nlayers 4 \
       --Tin $TIN \
       --Tout $TOUT \
       --checkpoint_params "${CHECKPOINT_PARAMS[@]}" \
       --log_params "${LOG_PARAMS[@]}" \
       --backbone_config_path "transformer_args.yaml" \
       --info "FL_Tout${TOUT}"

    echo "Finished experiment for TOUT=$TOUT."
    echo ""
done



