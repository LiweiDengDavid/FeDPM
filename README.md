# FL-VQVAE: Federated Learning with VQVAE for Time Series Forecasting

A PyTorch implementation for the project: **FL-VQVAE**. This project implements a **Federated Learning** framework combined with **VQVAE** for robust time-series forecasting.

## 🎯 Overview

This repository contains the implementation of a Federated Learning model utilizing Vector Quantized Variational AutoEncoders (VQVAE) to capture latent representations of time-series data across distributed clients.

## 📝 Install Dependencies

First, ensure you have the required packages installed.

```bash
pip install -r requirements.txt
```

## 👉 Data Preparation

### 1. Download Datasets
Please download the corresponding dataset files (e.g., `ETTh1`, `ETTh2`, `electricity`, `weather`, etc.).
Place them in the `Datasets` folder following this structure:

```text
Datasets/
└── imputation_and_forecasting_data/
    ├── electricity/
    ├── ETTh1/
    ├── ETTh2/
    └── ...
```

### 2. Data Preprocessing
Run the extraction script to generate `train`, `valid`, and `test` `.npy` files:
```bash
bash extract_all_data.sh
```
Processed data will be stored in `saved_data/`.

## 🚀 Run Experiment

We have provided experimental scripts for both **Full-Shot** and **Few-Shot** settings in the root directory.

### 1. Full-Shot Federated Learning
To execute the standard federated learning training:

```bash
bash train_FL_setting.sh
```

### 2. Few-Shot Learning (5% & 10%)
Perform experiments with limited data availability:

*   **5% Training Data**:
    
    ```bash
    bash few_shot_5percent.sh
    ```
    
*   **10% Training Data**:
    
    ```bash
    bash few_shot_10percent.sh
    ```

## ⚙️ Notes

*   **GPU Configuration**: Edit `cuda_id` or `BASE_CUDA_ID` in the respective shell scripts (`.sh`) to assign specific GPUs.
*   **Environment**: Python 3.x and PyTorch are required.

## 🌟 Citation

If you find this work is helpful to your research, please consider citing:

```
Coming Soon!
```
Thanks for your interest in our work!
