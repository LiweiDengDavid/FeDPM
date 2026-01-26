#!/bin/bash

# =====================================================================================
# Federated Learning Training Script for Multiple Forecasting Horizons (Tout)
# =====================================================================================

# Base configuration
BASE_CUDA_ID=1
BASE_SEED=2025
NUM_ROUNDS=100
LOCAL_EPOCHS=5
AGGREGATION_STRATEGY="cos_similarity"
CODEBOOK_FILL_STRATEGY="client_personalized"
CLIENT_DATA_TYPES=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "electricity" "exchange")
BASE_DATA_PATH="./saved_data/"
CONFIG_PATH="vqvae.json"
LOG_DIR="./logs/federated/"
BASE_LR=1e-5
D_MODEL=64
NHEAD=4
TIN=96

# Prediction horizons to iterate over
TOUT_VALUES=(96 192 336 720)

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
       --codebook_fill_strategy "$CODEBOOK_FILL_STRATEGY" \
       --client_data_types "${CLIENT_DATA_TYPES[@]}" \
       --base_data_path "$BASE_DATA_PATH" \
       --config_path "$CONFIG_PATH" \
       --log_dir "$LOG_DIR" \
       --baselr $BASE_LR \
       --d_model $D_MODEL \
       --nhead $NHEAD \
       --nlayers 4 \
       --Tin $TIN \
       --Tout $TOUT \
       --checkpoint_params "${CHECKPOINT_PARAMS[@]}" \
       --log_params "${LOG_PARAMS[@]}" \
       --ablation_config_path "Ablation_args_transformer.yaml" \
       --info "FL_Tout${TOUT}"

    echo "Finished experiment for TOUT=$TOUT."
    echo ""
done



