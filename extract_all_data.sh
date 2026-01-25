#!/bin/bash

# General settings
random_seed=2025
gpu=0
seq_len=96

# List of all datasets to process
# 
datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "electricity" "weather" "exchange" "traffic" "illness")

# Loop over each dataset
for model_id_name in ${datasets[@]}; do
  echo "--------------------------------------------------"
  echo "Processing dataset: $model_id_name"
  echo "--------------------------------------------------"

  # Default settings
  iter_seq_len=$seq_len
  iter_pred_lens="96 192 336 720"

  # Set dataset-specific parameters
  if [ "$model_id_name" = "ETTh1" ] || [ "$model_id_name" = "ETTh2" ] || [ "$model_id_name" = "ETTm1" ] || [ "$model_id_name" = "ETTm2" ]; then
    data_name=$model_id_name
    root_path_name="./Datasets/imputation_and_forecasting_data/ETT-small/"
    data_path_name=$model_id_name".csv"
    enc_in=7
  elif [ "$model_id_name" = "electricity" ]; then
    data_name="custom"
    root_path_name="./Datasets/imputation_and_forecasting_data/electricity/"
    data_path_name="electricity.csv"
    enc_in=321
  elif [ "$model_id_name" = "traffic" ]; then
    data_name="custom"
    root_path_name="./Datasets/imputation_and_forecasting_data/traffic/"
    data_path_name="traffic.csv"
    enc_in=862
  elif [ "$model_id_name" = "weather" ]; then
    data_name="custom"
    root_path_name="./Datasets/imputation_and_forecasting_data/weather/"
    data_path_name="weather.csv"
    enc_in=21
  elif [ "$model_id_name" = "exchange" ]; then
    data_name="custom"
    root_path_name="./Datasets/imputation_and_forecasting_data/exchange_rate/"
    data_path_name="exchange_rate.csv"
    enc_in=8
  elif [ "$model_id_name" = "illness" ]; then
    data_name="custom"
    root_path_name="./Datasets/imputation_and_forecasting_data/illness/"
    data_path_name="illness.csv"
    enc_in=7
    iter_seq_len=36
    iter_pred_lens="24 36 48 60"
  fi

  # Loop over each prediction length
  for pred_len in $iter_pred_lens; do
    echo " ==> Processing pred_len: $pred_len"

    python -u extract_forecasting_data.py \
      --random_seed $random_seed \
      --data $data_name \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --features M \
      --seq_len $iter_seq_len \
      --pred_len $pred_len \
      --label_len 0 \
      --enc_in $enc_in \
      --gpu $gpu \
      --save_path "./saved_data/$model_id_name/Tin${iter_seq_len}_Tout${pred_len}/" \
      --classifiy_or_forecast "forecast"
  done
done


echo "--------------------------------------------------"
echo "All datasets processed."
echo "--------------------------------------------------"
