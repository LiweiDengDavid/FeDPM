#!/bin/bash

# =====================================================================================
# Varied Length Ablation Script for Federated VQ-VAE
# =====================================================================================
# This script runs experiments with specific configurations for different prediction lengths:
# =====================================================================================

# Base configuration
BASE_CUDA_ID=0 # GPU ID
BASE_SEED=2025
NUM_ROUNDS=100
LOCAL_EPOCHS=5
AGGREGATION_STRATEGY="cos_similarity"
CLIENT_DATA_TYPES=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "electricity" "exchange")
BASE_DATA_PATH="./saved_data/"
LOG_DIR="./logs/sota/"
similarity_threshold=0.7
BASE_LR=1e-5
D_MODEL=64
NHEAD=4
TIN=96

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Checkpoint directory naming parameters
CHECKPOINT_PARAMS=("compression" "Tin" "Tout")
# Log filename parameters
LOG_PARAMS=("compression" "Tin" "Tout")

# =====================================================================================
# Function to generate YAML config
# =====================================================================================
generate_yaml_config() {
    local filename=$1
    local compression=$2
    local num_embeddings=$3
    local embedding_dim=$4

    cat <<EOF > "$filename"
vqvae_config:
    general_seed: 47
    model_name: "vqvae"
    model_save_name: "vqvae"
    pretrained: false
    learning_rate: 1e-3
    num_training_updates: 15000
    block_hidden_size: 128
    num_residual_layers: 2
    res_hidden_size: 64
    embedding_dim: $embedding_dim
    num_embeddings: $num_embeddings
    commitment_cost: 0.25
    compression_factor: $compression
comet_config:
    api_key: ""
    project_name: ""
    workspace: ""
EOF
}

# =====================================================================================
# Function to run a single experiment
# =====================================================================================
run_experiment() {
    local tout=$1
    local compression=$2
    local num_embeddings=$3
    local embedding_dim=$4
    local exp_name=$5

    echo "--------------------------------------------------------------------------------"
    echo "Running Experiment: $exp_name"
    echo "Tout: $tout | Compression: $compression | Num Embeddings: $num_embeddings | Dim: $embedding_dim"
    echo "--------------------------------------------------------------------------------"

    # Create temporary config file
    local config_file="temp_config_${exp_name}_T${tout}.yaml"
    generate_yaml_config "$config_file" "$compression" "$num_embeddings" "$embedding_dim"

    # Run the python script
    python federated_learning_main.py \
       --cuda-id $BASE_CUDA_ID \
       --seed $BASE_SEED \
       --num_rounds $NUM_ROUNDS \
       --local_epochs $LOCAL_EPOCHS \
       --aggregation_strategy "$AGGREGATION_STRATEGY" \
       --similarity_threshold $similarity_threshold \
       --client_data_types "${CLIENT_DATA_TYPES[@]}" \
       --base_data_path "$BASE_DATA_PATH" \
       --config_path "$config_file" \
       --log_dir "$LOG_DIR" \
       --baselr $BASE_LR \
       --d_model $D_MODEL \
       --nhead $NHEAD \
       --nlayers 4 \
       --Tin $TIN \
       --Tout $tout \
       --checkpoint_params "${CHECKPOINT_PARAMS[@]}" \
       --log_params "${LOG_PARAMS[@]}" \
       --ablation_config_path "Ablation_args_transformer.yaml" \
       --info "$exp_name"

    # Clean up
    rm "$config_file"
    
    echo "Experiment $exp_name (Tout=$tout) completed."
    echo ""
}

# =====================================================================================
# Main Experiment Loop
# =====================================================================================

# 1. Tout=96
# Requirement: patch len (compression) = 4 or 2, dimension = 64
# We use default num_embeddings = 256
run_experiment 96 4 256 64 "Tout96_Comp4_Dim64"
# run_experiment 96 2 256 64 "Tout96_Comp2_Dim64"

# 2. Tout=192
# Requirement: dimension = 64
# We use default compression = 4
run_experiment 192 4 256 64 "Tout192_Comp4_Dim64"


# 3. Tout=336
# Requirement: patch len = 4
# We use default dimension = 64 
for comp in 4; do
    run_experiment 336 $comp 256 64 "Tout336_Comp${comp}"
done


# 4. Tout=720
# Requirement: patch len = 4
# We use default dimension = 64
for comp in 4; do
    run_experiment 720 $comp 256 64 "Tout720_Comp${comp}"
done

echo "================================================================================================"
echo "All Varied Length Experiments Completed."
echo "================================================================================================"
