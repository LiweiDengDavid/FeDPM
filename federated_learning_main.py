import os
import json
import yaml
import argparse
from lib.federated_trainer import federated_train


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def default_argument_parser():
    parser = argparse.ArgumentParser(description="FeDPM Time Series Forecaster")
 
    # --- FeDPM Arguments ---
    parser.add_argument("--num_rounds", default=100, type=int, help="Number of communication rounds.")
    parser.add_argument("--local_epochs", default=5, type=int, help="Number of local training epochs on each client.")
    parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping.")
    parser.add_argument("--aggregation_strategy", default='cos_similarity', type=str, choices=['fedavg', 'cos_similarity'], help="Federated aggregation strategy.")
    parser.add_argument("--similarity_threshold", default=0.7, type=float, help="Similarity threshold for cosine similarity aggregation (only used when aggregation_strategy='cos_similarity').")
    parser.add_argument("--gamma", default=0.95, type=float, help="Maximum ratio of shared global memory vectors. The remaining slots are filled based on memory_fill_strategy.")
    parser.add_argument("--memory_fill_strategy", default='client_personalized', type=str, choices=['random_isolated', 'client_personalized', 'none'], 
                       help="Strategy to fill remaining memory slots when aggregated clusters are insufficient. "
                       "'random_isolated': randomly select from isolated vectors (global); "
                       "'client_personalized': each client gets personalized vectors (lowest cross-client similarity); "
                       "'none': leave remaining slots as zeros.")
    parser.add_argument("--disable_memory_aggregation", action='store_true', help="If set, clients keep their own local memories without aggregation (ablation study).")
    parser.add_argument("--dp_noise_type", default='none', type=str, choices=['none', 'laplace', 'gaussian', 'exponential'], 
                       help="Type of differential privacy noise to add to memory uploads. "
                       "'none': no noise, 'laplace': Laplace noise (μ=0, λ=1), "
                       "'gaussian': Gaussian noise (μ=0, σ=1), 'exponential': Exponential noise (λ=1).")
    parser.add_argument("--dp_noise_scale", default=1.0, type=float, 
                       help="Noise scale factor for differential privacy. Higher values add more noise (stronger privacy).")
    parser.add_argument("--client_data_types", nargs='+', default=["ETTh1", "ETTh2"], help="List of datasets, one for each client.")
    parser.add_argument("--base_data_path", default='./saved_data/', type=str, help="Base path where client datasets are stored in subfolders.")
    parser.add_argument("--checkpoint_path", default='./saved_models/', type=str, help="Base directory to save the best model checkpoints.")
    
    # --- Few-shot and Zero-shot Arguments ---
    parser.add_argument("--few_shot_ratio", default=1.0, type=float, help="Ratio of training data to use for few-shot learning (e.g., 0.05 for 5%%, 0.1 for 10%%). Default 1.0 (full data).")
    parser.add_argument("--zero_shot_clients", nargs='+', default=[], help="List of client dataset names that should NOT participate in training (zero-shot clients). They will only be evaluated.")
   

    # --- General Arguments ---
    parser.add_argument("--cuda-id", default=0, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--backbone_config_path", default="transformer_args.yaml", type=str, help="Path to the ablation configuration YAML file.")
    parser.add_argument("--log_dir", default='./logs', type=str, help="Directory to save logs.")
    parser.add_argument("--results_path", default='FeDPM_results.jsonl', type=str, help="Path to the JSON Lines file for logging experiment results.")
    
    # --- Data Arguments ---
    parser.add_argument("--Tin", default=96, type=int)
    parser.add_argument("--Tout", default=96, type=int)

    # --- Training Arguments ---
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--baselr", default=1e-5, type=float)
    parser.add_argument("--loss_type", default="smoothl1", type=str)
    parser.add_argument("--beta", default=0.1, type=float, help="beta for smoothl1 loss")
    parser.add_argument("--onehot", action="store_true", help="Use one-hot for code representation.")
    parser.add_argument("--scheme", default=1, type=int, help="1 predicts mu/std, 2 uses revin denorm.")
    parser.add_argument("--clients_disable_mustd", nargs='+', default=[], help="List of client dataset names that should NOT use the MuStdModel.")

    # --- Model Architecture Arguments ---
    parser.add_argument("--d_model", default=64, type=int)
    parser.add_argument("--d_hid", default=256, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--nlayers", default=4, type=int)
    
    # --- Encoder & Retrieval Arguments ---
    parser.add_argument("--block_hidden_size", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--res_hidden_size", default=64, type=int)
    parser.add_argument("--embedding_dim", default=64, type=int, help="Dimension of each embedding vector.")
    parser.add_argument("--num_embeddings", default=256, type=int, help="Memory Size.")
    parser.add_argument("--commitment_cost", default=0.25, type=float, help="Follow the VQ-VAE.")
    parser.add_argument("--compression_factor", default=4, type=int,help="Patch Length")

    # --- Checkpoint Parameters Configuration ---
    parser.add_argument("--checkpoint_params", nargs='+', default=['Tin', 'Tout', 'seed'], 
                       type=str, help="Hyperparameters to include in checkpoint directory name. "
                       "Available options: Tin, Tout, Seed, Rounds, Epochs, LR, DModel, NHead, NLayers, Compression and others. ")

    # --- Log Parameters Configuration ---
    parser.add_argument("--log_params", nargs='+', default=['num_rounds', 'local_epochs',"Tin",'seed'],
                       type=str, help="Hyperparameters to include in log filename. "
                       "Available options: num_rounds, local_epochs, seed, baselr, d_model, nhead, nlayers, compression, Tin, Tout and others. ")

    parser.add_argument("--info", required=False, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    federated_train(args)
