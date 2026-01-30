#!/bin/bash

# Few-shot Learning Experiment: 5% Training Data
# This script runs federated learning with only 5% of the training data for each client

# General settings
CUDA_ID=1
SEED=2025
NUM_ROUNDS=100
LOCAL_EPOCHS=5
PATIENCE=10

# Few-shot setting
FEW_SHOT_RATIO=0.05  # 5% of training data

# Data settings
TIN=96
TOUT=96

# Aggregation settings
AGGREGATION_STRATEGY="cos_similarity"
SIMILARITY_THRESHOLD=0.7
GAMMA=0.95
MEMORY_FILL_STRATEGY="client_personalized"

# Paths
backbone_config_path="transformer_args.yaml"
LOG_DIR="./logs/federated_few_shot"
RESULTS_PATH="results_few_shot_5percent.jsonl"
CHECKPOINT_PATH="./saved_models/"
BASE_DATA_PATH="./saved_data/"

# Client datasets - modify as needed
CLIENT_DATA_TYPES=("ETTm1" "ETTm2" "electricity" "weather" "exchange")
CLIENTS_DISABLE_MUSTD=("ETTm1" "ETTm2")
echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${CLIENT_DATA_TYPES[@]}"
echo "Few-shot ratio: ${FEW_SHOT_RATIO} (5%)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Local epochs: ${LOCAL_EPOCHS}"
echo "Seed: ${SEED}"
echo "=================================================="

python federated_learning_main.py \
    --cuda-id ${CUDA_ID} \
    --seed ${SEED} \
    --num_rounds ${NUM_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --patience ${PATIENCE} \
    --few_shot_ratio ${FEW_SHOT_RATIO} \
    --aggregation_strategy ${AGGREGATION_STRATEGY} \
    --similarity_threshold ${SIMILARITY_THRESHOLD} \
    --gamma ${GAMMA} \
    --memory_fill_strategy ${MEMORY_FILL_STRATEGY} \
    --client_data_types ${CLIENT_DATA_TYPES[@]} \
    --clients_disable_mustd ${CLIENTS_DISABLE_MUSTD[@]} \
    --Tin ${TIN} \
    --Tout ${TOUT} \
    --config_path ${CONFIG_PATH} \
    --backbone_config_path ${backbone_config_path} \
    --log_dir ${LOG_DIR} \
    --results_path ${RESULTS_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --base_data_path ${BASE_DATA_PATH} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${RESULTS_PATH}"
echo "=================================================="



echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${CLIENT_DATA_TYPES[@]}"
echo "Few-shot ratio: ${FEW_SHOT_RATIO} (5%)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Local epochs: ${LOCAL_EPOCHS}"
echo "Seed: ${SEED}"
echo "=================================================="
TOUT=192
python federated_learning_main.py \
    --cuda-id ${CUDA_ID} \
    --seed ${SEED} \
    --num_rounds ${NUM_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --patience ${PATIENCE} \
    --few_shot_ratio ${FEW_SHOT_RATIO} \
    --aggregation_strategy ${AGGREGATION_STRATEGY} \
    --similarity_threshold ${SIMILARITY_THRESHOLD} \
    --gamma ${GAMMA} \
    --memory_fill_strategy ${MEMORY_FILL_STRATEGY} \
    --client_data_types ${CLIENT_DATA_TYPES[@]} \
    --clients_disable_mustd ${CLIENTS_DISABLE_MUSTD[@]} \
    --Tin ${TIN} \
    --Tout ${TOUT} \
    --config_path ${CONFIG_PATH} \
    --backbone_config_path ${backbone_config_path} \
    --log_dir ${LOG_DIR} \
    --results_path ${RESULTS_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --base_data_path ${BASE_DATA_PATH} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${RESULTS_PATH}"
echo "=================================================="



echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${CLIENT_DATA_TYPES[@]}"
echo "Few-shot ratio: ${FEW_SHOT_RATIO} (5%)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Local epochs: ${LOCAL_EPOCHS}"
echo "Seed: ${SEED}"
echo "=================================================="
TOUT=336
CLIENT_DATA_TYPES=("ETTm1" "ETTm2" "electricity" "weather")
python federated_learning_main.py \
    --cuda-id ${CUDA_ID} \
    --seed ${SEED} \
    --num_rounds ${NUM_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --patience ${PATIENCE} \
    --few_shot_ratio ${FEW_SHOT_RATIO} \
    --aggregation_strategy ${AGGREGATION_STRATEGY} \
    --similarity_threshold ${SIMILARITY_THRESHOLD} \
    --gamma ${GAMMA} \
    --memory_fill_strategy ${MEMORY_FILL_STRATEGY} \
    --client_data_types ${CLIENT_DATA_TYPES[@]} \
    --clients_disable_mustd ${CLIENTS_DISABLE_MUSTD[@]} \
    --Tin ${TIN} \
    --Tout ${TOUT} \
    --config_path ${CONFIG_PATH} \
    --backbone_config_path ${backbone_config_path} \
    --log_dir ${LOG_DIR} \
    --results_path ${RESULTS_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --base_data_path ${BASE_DATA_PATH} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${RESULTS_PATH}"
echo "=================================================="


echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${CLIENT_DATA_TYPES[@]}"
echo "Few-shot ratio: ${FEW_SHOT_RATIO} (5%)"
echo "Rounds: ${NUM_ROUNDS}"
echo "Local epochs: ${LOCAL_EPOCHS}"
echo "Seed: ${SEED}"
echo "=================================================="
TOUT=720
python federated_learning_main.py \
    --cuda-id ${CUDA_ID} \
    --seed ${SEED} \
    --num_rounds ${NUM_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --patience ${PATIENCE} \
    --few_shot_ratio ${FEW_SHOT_RATIO} \
    --aggregation_strategy ${AGGREGATION_STRATEGY} \
    --similarity_threshold ${SIMILARITY_THRESHOLD} \
    --gamma ${GAMMA} \
    --memory_fill_strategy ${MEMORY_FILL_STRATEGY} \
    --client_data_types ${CLIENT_DATA_TYPES[@]} \
    --clients_disable_mustd ${CLIENTS_DISABLE_MUSTD[@]} \
    --Tin ${TIN} \
    --Tout ${TOUT} \
    --config_path ${CONFIG_PATH} \
    --backbone_config_path ${backbone_config_path} \
    --log_dir ${LOG_DIR} \
    --results_path ${RESULTS_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --base_data_path ${BASE_DATA_PATH} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${RESULTS_PATH}"
echo "=================================================="