#!/bin/bash

# Few-shot Learning Experiment: 5% Training Data
# This script runs federated learning with only 5% of the training data for each client

# General settings
cuda_id=1
seed=2025
num_rounds=100
local_epochs=5
patience=10

# Few-shot setting
few_shot_ratio=0.05  # 5% of training data

# Data settings
Tin=96
Tout=96

# Aggregation settings
aggregation_strategy="cos_similarity"
similarity_threshold=0.7
gamma=0.95
codebook_fill_strategy="client_personalized"

# Paths
config_path="vqvae.json"
ablation_config_path="Ablation_args_transformer.yaml"
log_dir="./logs/federated_few_shot"
results_path="results_few_shot_5percent.jsonl"
checkpoint_path="./saved_models/"
base_data_path="./saved_data/"

# Client datasets - modify as needed
client_data_types=("ETTm1" "ETTm2" "electricity" "weather" "exchange")
clients_disable_mustd=("ETTm1" "ETTm2")
echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${client_data_types[@]}"
echo "Few-shot ratio: ${few_shot_ratio} (5%)"
echo "Rounds: ${num_rounds}"
echo "Local epochs: ${local_epochs}"
echo "Seed: ${seed}"
echo "=================================================="

python federated_learning_main.py \
    --cuda-id ${cuda_id} \
    --seed ${seed} \
    --num_rounds ${num_rounds} \
    --local_epochs ${local_epochs} \
    --patience ${patience} \
    --few_shot_ratio ${few_shot_ratio} \
    --aggregation_strategy ${aggregation_strategy} \
    --similarity_threshold ${similarity_threshold} \
    --gamma ${gamma} \
    --codebook_fill_strategy ${codebook_fill_strategy} \
    --client_data_types ${client_data_types[@]} \
    --clients_disable_mustd ${clients_disable_mustd[@]} \
    --Tin ${Tin} \
    --Tout ${Tout} \
    --config_path ${config_path} \
    --ablation_config_path ${ablation_config_path} \
    --log_dir ${log_dir} \
    --results_path ${results_path} \
    --checkpoint_path ${checkpoint_path} \
    --base_data_path ${base_data_path} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${results_path}"
echo "=================================================="



echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${client_data_types[@]}"
echo "Few-shot ratio: ${few_shot_ratio} (5%)"
echo "Rounds: ${num_rounds}"
echo "Local epochs: ${local_epochs}"
echo "Seed: ${seed}"
echo "=================================================="
Tout=192
python federated_learning_main.py \
    --cuda-id ${cuda_id} \
    --seed ${seed} \
    --num_rounds ${num_rounds} \
    --local_epochs ${local_epochs} \
    --patience ${patience} \
    --few_shot_ratio ${few_shot_ratio} \
    --aggregation_strategy ${aggregation_strategy} \
    --similarity_threshold ${similarity_threshold} \
    --gamma ${gamma} \
    --codebook_fill_strategy ${codebook_fill_strategy} \
    --client_data_types ${client_data_types[@]} \
    --clients_disable_mustd ${clients_disable_mustd[@]} \
    --Tin ${Tin} \
    --Tout ${Tout} \
    --config_path ${config_path} \
    --ablation_config_path ${ablation_config_path} \
    --log_dir ${log_dir} \
    --results_path ${results_path} \
    --checkpoint_path ${checkpoint_path} \
    --base_data_path ${base_data_path} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${results_path}"
echo "=================================================="



echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${client_data_types[@]}"
echo "Few-shot ratio: ${few_shot_ratio} (5%)"
echo "Rounds: ${num_rounds}"
echo "Local epochs: ${local_epochs}"
echo "Seed: ${seed}"
echo "=================================================="
Tout=336
client_data_types=("ETTm1" "ETTm2" "electricity" "weather")
python federated_learning_main.py \
    --cuda-id ${cuda_id} \
    --seed ${seed} \
    --num_rounds ${num_rounds} \
    --local_epochs ${local_epochs} \
    --patience ${patience} \
    --few_shot_ratio ${few_shot_ratio} \
    --aggregation_strategy ${aggregation_strategy} \
    --similarity_threshold ${similarity_threshold} \
    --gamma ${gamma} \
    --codebook_fill_strategy ${codebook_fill_strategy} \
    --client_data_types ${client_data_types[@]} \
    --clients_disable_mustd ${clients_disable_mustd[@]} \
    --Tin ${Tin} \
    --Tout ${Tout} \
    --config_path ${config_path} \
    --ablation_config_path ${ablation_config_path} \
    --log_dir ${log_dir} \
    --results_path ${results_path} \
    --checkpoint_path ${checkpoint_path} \
    --base_data_path ${base_data_path} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${results_path}"
echo "=================================================="


echo "=================================================="
echo "Few-shot Learning Experiment: 5% Training Data"
echo "=================================================="
echo "Clients: ${client_data_types[@]}"
echo "Few-shot ratio: ${few_shot_ratio} (5%)"
echo "Rounds: ${num_rounds}"
echo "Local epochs: ${local_epochs}"
echo "Seed: ${seed}"
echo "=================================================="
Tout=720
python federated_learning_main.py \
    --cuda-id ${cuda_id} \
    --seed ${seed} \
    --num_rounds ${num_rounds} \
    --local_epochs ${local_epochs} \
    --patience ${patience} \
    --few_shot_ratio ${few_shot_ratio} \
    --aggregation_strategy ${aggregation_strategy} \
    --similarity_threshold ${similarity_threshold} \
    --gamma ${gamma} \
    --codebook_fill_strategy ${codebook_fill_strategy} \
    --client_data_types ${client_data_types[@]} \
    --clients_disable_mustd ${clients_disable_mustd[@]} \
    --Tin ${Tin} \
    --Tout ${Tout} \
    --config_path ${config_path} \
    --ablation_config_path ${ablation_config_path} \
    --log_dir ${log_dir} \
    --results_path ${results_path} \
    --checkpoint_path ${checkpoint_path} \
    --base_data_path ${base_data_path} \
    --checkpoint_params Tin Tout few_shot_ratio \
    --log_params num_rounds local_epochs Tin few_shot_ratio

echo "=================================================="
echo "Few-shot 5% experiment completed!"
echo "Results saved to: ${results_path}"
echo "=================================================="