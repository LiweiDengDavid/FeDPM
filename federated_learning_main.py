import torch
import numpy as np
import os
import json
import yaml
import argparse
from collections import OrderedDict
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng
from lib.utils.logger import get_color_logger as get_logger
from lib.utils.results_logger import log_results_to_jsonl
from federated_components import Server, Client
from lib.utils.load_args import load_additional_config, merge_args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# #############################################################################
# Main FeDPM Orchestration
# #############################################################################

def build_checkpoint_dirname(args):
    """
    Build the checkpoint directory name based on the configured hyperparameters.
    
    Args:
        args: Arguments object containing checkpoint_params and hyperparameter values
        
    Returns:
        str: Directory name in format like "Federated_Tin96_Tout96_Seed2025"
    """
    # Default parameters if not specified
    checkpoint_params = getattr(args, 'checkpoint_params', ['Tin', 'Tout', 'Seed'])
    
    # Build the directory name from specified parameters
    param_parts = []
    for param in checkpoint_params:
        attr_name = param
        if hasattr(args, attr_name):
            value = getattr(args, attr_name)
            param_parts.append(f"{param}{value}")
    
    if param_parts:
        dirname = "FeDPM_" + "_".join(param_parts)
    else:
        # Fallback if no valid parameters specified
        dirname = f"FeDPM_Tin{args.Tin}_Tout{args.Tout}_Seed{args.seed}"
    
    return dirname


def build_log_filename(args):
    """
    Build the log filename based on the configured hyperparameters.
    
    Args:
        args: Arguments object containing log_params and hyperparameter values
        
    Returns:
        str: Log filename in format like "FL_Tin96_Tout96_Seed2025.log"
    """
    # Default parameters if not specified
    log_params = getattr(args, 'log_params', ['num_rounds', 'local_epochs', 'seed'])
    
    # Build the log filename from specified parameters
    param_parts = []
    for param in log_params:
        if hasattr(args, param):
            value = getattr(args, param)
            param_parts.append(f"{param}{value}")
    
    if param_parts:
        log_filename = "FeDPM_" + "_".join(param_parts) + ".log"
    else:
        # Fallback if no valid parameters specified
        log_filename = f"FeDPM_rounds{args.num_rounds}_epochs{args.local_epochs}_seed{args.seed}.log"
    
    return log_filename


def add_noise_to_memory(memory, noise_type, noise_scale, device):
    """
    Add differential privacy noise to memory tensor.
    
    Args:
        memory: Tensor of shape (num_embeddings, embedding_dim)
        noise_type: Type of noise ('laplace', 'gaussian', 'exponential', or 'none')
        noise_scale: Scale factor for noise (privacy budget)
        device: Device to create noise tensors on
        
    Returns:
        Noisy memory tensor with same shape as input
    """
    if noise_type == 'none':
        return memory
    
    num_embeddings, embedding_dim = memory.shape
    
    if noise_type == 'laplace':
        # Laplace distribution: μ=0, λ=1
        # In PyTorch, Laplace is parameterized by (loc, scale) where scale = 1/λ
        # So for λ=1, we use scale=1/noise_scale
        laplace_dist = torch.distributions.Laplace(0, 1.0 / noise_scale)
        noise = laplace_dist.sample(memory.shape).to(device)
        
    elif noise_type == 'gaussian':
        # Gaussian distribution: μ=0, σ=1
        # In PyTorch, Normal is parameterized by (mean, std)
        # So for σ=1, we use std=noise_scale
        gaussian_dist = torch.distributions.Normal(0, noise_scale)
        noise = gaussian_dist.sample(memory.shape).to(device)
        
    elif noise_type == 'exponential':
        # Exponential distribution: λ=1
        # In PyTorch, Exponential is parameterized by rate (λ)
        # So for λ=1, we use rate=1/noise_scale
        exponential_dist = torch.distributions.Exponential(1.0 / noise_scale)
        noise = exponential_dist.sample(memory.shape).to(device)
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    noisy_memory = memory + noise
    return noisy_memory


def federated_train(args):
    """
    Main function to orchestrate the Federated Learning process.
    """
    # Load ablation configuration from YAML if available
    if hasattr(args, 'backbone_config_path') and args.backbone_config_path:
        if os.path.exists(args.backbone_config_path):
            yaml_args = load_additional_config(args.backbone_config_path)
            args = merge_args(args, yaml_args)
        else:
            raise FileNotFoundError(f"Ablation config file not found: {args.backbone_config_path}")
    
    # Set seed for reproducibility
    seed_all_rng(None if args.seed < 0 else args.seed)
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")

    # Initialize Logger with dynamically built filename
    log_filename = build_log_filename(args)
    logger = get_logger(log_dir=args.log_dir, name='FeDPM', log_filename=log_filename)
 
    # Log arguments
    logger.info(f"|{'=' * 101}|")
    logger.info(f"| {'FeDPM Configuration':^99} |")
    logger.info(f"|{'=' * 101}|")
    for key, value in args.__dict__.items():
        logger.info(f"|{str(key):>50s}|{str(value):<50s}|")
    logger.info(f"|{'=' * 101}|")
    logger.info(f"|{'=' * 101}|")

    # 1. Initialize Server
    server = Server(args, device, logger, aggregation_strategy=args.aggregation_strategy, 
                   similarity_threshold=args.similarity_threshold,
                   memory_fill_strategy=args.memory_fill_strategy,gamma=args.gamma)

    # 2. Initialize Clients
    clients = [Client(i, args, device, logger) for i in range(len(args.client_data_types))]
    
    # Separate clients into training and zero-shot groups
    training_clients = []
    zero_shot_clients = []
    
    for client in clients:
        if client.data_type in args.zero_shot_clients:
            zero_shot_clients.append(client)
            logger.info(f"Client {client.client_id} ({client.data_type}) marked as ZERO-SHOT (evaluation only).")
        else:
            training_clients.append(client)
            logger.info(f"Client {client.client_id} ({client.data_type}) marked as TRAINING client.")
    
    if len(training_clients) == 0:
        raise ValueError("No training clients available! All clients are marked as zero-shot.")
    
    logger.info(f"Total clients: {len(clients)} | Training: {len(training_clients)} | Zero-shot: {len(zero_shot_clients)}")
    
    # Initialize Early Stopping
    data_list = "-".join(args.client_data_types)
    checkpoint_dirname = build_checkpoint_dirname(args)
    checkpoint_path = os.path.join(args.checkpoint_path, f'{data_list}', checkpoint_dirname)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Checkpoint parameters: {getattr(args, 'checkpoint_params', ['Tin', 'Tout', 'Seed'])}")
    os.makedirs(checkpoint_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, path=checkpoint_path)
    
    # 3. Start Federated Training Rounds
    for round_num in range(args.num_rounds):
        logger.info(f"\n{'='*40} ROUND {round_num + 1}/{args.num_rounds} {'='*40}")
        
        # Select only training clients for this round (zero-shot clients don't train)
        selected_clients = training_clients

        client_updates = []
        client_data_sizes = []
        client_usage_counts = []  # Track memory usage frequency

        # Get current global model weights from the server
        global_weights = server.get_global_weights()

        # Broadcast to clients and perform local training
        for client in selected_clients:
            updated_memory = client.local_train(global_weights)
            if updated_memory is not None:
                client_updates.append(updated_memory)
                client_data_sizes.append(client.data_size)
                # Collect usage frequency statistics
                client_usage_counts.append(client.memory_usage_count.clone())
        
        # Add differential privacy noise to memory uploads if enabled
        if args.dp_noise_type != 'none':
            logger.info(f"Applying {args.dp_noise_type} noise (scale={args.dp_noise_scale}) to memory uploads...")
            noisy_client_updates = []
            for i, memory in enumerate(client_updates):
                noisy_memory = add_noise_to_memory(
                    memory, 
                    args.dp_noise_type, 
                    args.dp_noise_scale, 
                    device
                )
                noisy_client_updates.append(noisy_memory)
                logger.info(f"  Client {selected_clients[i].client_id}: Added {args.dp_noise_type} noise to memory (shape={memory.shape})")
            client_updates = noisy_client_updates
        
        # Aggregate updates on the server (skip if memory aggregation is disabled)
        if client_updates and not args.disable_memory_aggregation:
            # Store client_updates for potential best model visualization
            # We'll reuse them if this round becomes the best
            current_round_updates = client_updates.copy()
            
            personalized_memories = server.aggregate(
                client_updates, 
                client_data_sizes, 
                client_usage_counts=client_usage_counts
            )
        elif args.disable_memory_aggregation:
            logger.info("Memory aggregation is disabled. Clients keep their own local memories.")
            current_round_updates = None
            personalized_memories = None
        else:
            logger.warning("No client updates received in this round. Skipping aggregation.")
            current_round_updates = None
            personalized_memories = None

        # --- Federated Validation for Early Stopping ---
        # NOTE: Validation only uses TRAINING clients for early stopping decision
        # Zero-shot clients are evaluated separately after training
        val_metrics_all_clients = {'mse': [], 'mae': []}
        
        # When memory aggregation is disabled, use each client's own local memory for validation
        if args.disable_memory_aggregation:
            for client in training_clients:  # Only validate training clients
                # Each client evaluates using its own local memory (no global weights needed)
                val_metrics = client.evaluate(None, split='val', use_local_memory=True)
                if val_metrics:
                    val_metrics_all_clients['mse'].append(val_metrics['mse'])
                    val_metrics_all_clients['mae'].append(val_metrics['mae'])
        else:
            # Normal federated setting: check if we have personalized memories
            if personalized_memories is not None:
                # client_personalized strategy: each client uses its own personalized memory
                logger.info("Using personalized memories for validation.")
                for i, client in enumerate(training_clients):  # Only validate training clients
                    val_metrics = client.evaluate(personalized_memories[i], split='val')
                    if val_metrics:
                        val_metrics_all_clients['mse'].append(val_metrics['mse'])
                        val_metrics_all_clients['mae'].append(val_metrics['mae'])
            else:
                # Standard federated: all clients use the same global memory
                current_global_weights = server.get_global_weights()
                for client in training_clients:  # Only validate training clients
                    val_metrics = client.evaluate(current_global_weights, split='val')
                    if val_metrics:
                        val_metrics_all_clients['mse'].append(val_metrics['mse'])
                        val_metrics_all_clients['mae'].append(val_metrics['mae'])
        
        if val_metrics_all_clients['mse']:
            avg_val_mse = np.mean(val_metrics_all_clients['mse'])
            avg_val_mae = np.mean(val_metrics_all_clients['mae'])
            logger.info(f"Round {round_num + 1} | Average Validation MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}")
            
            # Collect all models for checkpointing (only training clients)
            models_to_save = {'global_model': server.global_model}
            for client in training_clients:
                models_to_save.update(client.get_local_models())

            # Early stopping call uses mse, mae, and a dictionary of all models
            early_stopping(avg_val_mse, avg_val_mae, models_to_save)

            # If a new best model was saved, also save the global memory and visualization
            if early_stopping.counter == 0:
                # Only save global memory if aggregation is enabled
                if not args.disable_memory_aggregation:
                    # Check if we have personalized memories
                    if personalized_memories is not None:
                        # Save personalized memory for each training client
                        for i, client in enumerate(training_clients):
                            personalized_memory = personalized_memories[i]['_embedding.weight'].detach().cpu().numpy()
                            memory_path = os.path.join(early_stopping.path, f'memory_client_{client.client_id}_{client.data_type}_personalized.npy')
                            np.save(memory_path, personalized_memory)
                        logger.info(f"Saved personalized memories for all training clients.")
                    else:
                        # Save standard global memory
                        memory = server.global_model._embedding.weight.detach().cpu().numpy()
                        memory_path = os.path.join(early_stopping.path, 'memory.npy')
                        np.save(memory_path, memory)
                        logger.info(f"Saved new best memory to {memory_path}")
                else:
                    # When aggregation is disabled, save each training client's local memory
                    for client in training_clients:
                        local_memory = client.model_vq._embedding.weight.detach().cpu().numpy()
                        local_memory_path = os.path.join(early_stopping.path, f'memory_client_{client.client_id}_{client.data_type}.npy')
                        np.save(local_memory_path, local_memory)
                    logger.info(f"Saved local memories for all training clients (no aggregation mode)")
            
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

    logger.info("\nFeDPM process finished.")
    
    # --- Final Evaluation on Test Set ---
    logger.info(f"\n{'='*40} FINAL EVALUATION {'='*40}")
    # Load the best models saved by early stopping
    
    # Load global model (only if aggregation is enabled)
    if not args.disable_memory_aggregation:
        best_global_model_path = os.path.join(early_stopping.path, 'global_model_checkpoint.pth')
        logger.info(f"Loading best global model from: {best_global_model_path}")
        if os.path.exists(best_global_model_path):
            server.global_model.load_state_dict(torch.load(best_global_model_path))
        else:
            logger.error(f"Could not find best global model at path: {best_global_model_path}. Using last-round model for final evaluation.")
    else:
        logger.info("Memory aggregation disabled - clients use their own local memories for evaluation.")

    # Load local models for each training client
    for client in training_clients:
        local_models = client.get_local_models()
        for model_name, model_instance in local_models.items():
            model_path = os.path.join(early_stopping.path, f'{model_name}_checkpoint.pth')
            logger.info(f"Loading best local model for Client {client.client_id} ({model_name}) from: {model_path}")
            if os.path.exists(model_path):
                model_instance.load_state_dict(torch.load(model_path))
            else:
                logger.error(f"Could not find best local model at path: {model_path}. Using last-round model for final evaluation.")


    final_global_weights = server.get_global_weights() if not args.disable_memory_aggregation else None
    
    # Check if we need to load personalized memories for final evaluation
    has_personalized_memories = False
    if not args.disable_memory_aggregation and args.memory_fill_strategy == 'client_personalized':
        has_personalized_memories = True
        logger.info("Loading personalized memories for final evaluation.")
    
    # --- Evaluate Training Clients ---
    all_client_metrics = {}
    logger.info(f"\n{'='*20} Evaluating TRAINING Clients {'='*20}")
    for client in training_clients:
        if args.disable_memory_aggregation:
            # Each client uses its own local memory
            test_metrics = client.evaluate(None, split='test', use_local_memory=True)
        elif has_personalized_memories:
            # Load the personalized memory for this client
            personalized_memory_path = os.path.join(early_stopping.path, f'memory_client_{client.client_id}_{client.data_type}_personalized.npy')
            if os.path.exists(personalized_memory_path):
                personalized_memory_np = np.load(personalized_memory_path)
                personalized_memory_tensor = torch.from_numpy(personalized_memory_np).to(device)
                personalized_weights = OrderedDict()
                personalized_weights['_embedding.weight'] = personalized_memory_tensor
                test_metrics = client.evaluate(personalized_weights, split='test')
                logger.info(f"Client {client.client_id}: Using personalized memory from {personalized_memory_path}")
            else:
                logger.warning(f"Client {client.client_id}: Personalized memory not found, using global memory.")
                test_metrics = client.evaluate(final_global_weights, split='test')
        else:
            # Use the global memory
            test_metrics = client.evaluate(final_global_weights, split='test')
            
        if test_metrics:
            all_client_metrics[client.data_type] = test_metrics
            logger.info(f"Training Client {client.client_id} ({client.data_type}) Test Metrics: "
                        f"MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}, Corr={test_metrics['corr']:.4f}")
    
    # --- Evaluate Zero-shot Clients ---
    if len(zero_shot_clients) > 0:
        logger.info(f"\n{'='*20} Evaluating ZERO-SHOT Clients {'='*20}")
        
        for zero_client in zero_shot_clients:
            logger.info(f"\nZero-shot client {zero_client.client_id} ({zero_client.data_type}): "
                       f"Testing with each training client's model and memory...")
            
            best_mse = float('inf')
            best_metrics = None
            best_source_client = None
            
            # Try each training client's model
            for train_client in training_clients:
                logger.info(f"  Testing with Training Client {train_client.client_id} ({train_client.data_type}) model...")
                
                # Load training client's encoder and decoder weights to zero-shot client
                zero_client.model_encoder.load_state_dict(train_client.model_encoder.state_dict())
                zero_client.model_decode.load_state_dict(train_client.model_decode.state_dict())
                
                # Load mustd model if it exists
                if zero_client.model_mustd is not None and train_client.model_mustd is not None:
                    zero_client.model_mustd.load_state_dict(train_client.model_mustd.state_dict())
                
                # Determine which memory to use based on the strategy
                if args.disable_memory_aggregation:
                    # Use training client's local memory
                    zero_client.model_vq.load_state_dict(train_client.model_vq.state_dict())
                    test_metrics = zero_client.evaluate(None, split='test', use_local_memory=True)
                    
                elif has_personalized_memories:
                    # Use training client's personalized memory
                    personalized_memory_path = os.path.join(early_stopping.path, 
                                                             f'memory_client_{train_client.client_id}_{train_client.data_type}_personalized.npy')
                    
                    if os.path.exists(personalized_memory_path):
                        personalized_memory_np = np.load(personalized_memory_path)
                        personalized_memory_tensor = torch.from_numpy(personalized_memory_np).to(device)
                        personalized_weights = OrderedDict()
                        personalized_weights['_embedding.weight'] = personalized_memory_tensor
                        test_metrics = zero_client.evaluate(personalized_weights, split='test')
                    else:
                        logger.warning(f"Personalized memory not found for training client {train_client.client_id}")
                        continue
                        
                else:
                    # Use global memory
                    test_metrics = zero_client.evaluate(final_global_weights, split='test')
                
                # Track the best performing training client
                if test_metrics and test_metrics['mse'] < best_mse:
                    best_mse = test_metrics['mse']
                    best_metrics = test_metrics
                    best_source_client = train_client
                    logger.info(f"    -> New best MSE: {best_mse:.4f}")
            
            # Log the best result for this zero-shot client
            if best_metrics:
                all_client_metrics[zero_client.data_type] = best_metrics
                logger.info(f"\n✓ Zero-shot Client {zero_client.client_id} ({zero_client.data_type}) | "
                          f"Best source: Training Client {best_source_client.client_id} ({best_source_client.data_type}) | "
                          f"MSE={best_metrics['mse']:.4f}, MAE={best_metrics['mae']:.4f}, Corr={best_metrics['corr']:.4f}")
            else:
                logger.error(f"✗ Zero-shot Client {zero_client.client_id} ({zero_client.data_type}): No valid metrics obtained!")

    # Calculate and log average metrics across all clients (training + zero-shot)
    avg_metrics = {'mse': [], 'mae': [], 'corr': []}
    for metrics in all_client_metrics.values():
        avg_metrics['mse'].append(metrics['mse'])
        avg_metrics['mae'].append(metrics['mae'])
        avg_metrics['corr'].append(metrics['corr'])

    if avg_metrics['mse']:
        final_avg_mse = np.mean(avg_metrics['mse'])
        final_avg_mae = np.mean(avg_metrics['mae'])
        final_avg_corr = np.mean(avg_metrics['corr'])
        
        logger.info(f"\nAverage Test Metrics Across All Clients: "
                    f"MSE={final_avg_mse:.4f}, "
                    f"MAE={final_avg_mae:.4f}, "
                    f"Corr={final_avg_corr:.4f}")
        
        # --- Log final results to JSONL file ---
        # Log individual client metrics instead of the average
        log_results_to_jsonl(args.results_path, args, all_client_metrics)
        logger.info(f"Full experiment results have been logged to {args.results_path}")

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
                       "Available options: Tin, Tout, Seed, Rounds, Epochs, LR, DModel, NHead, NLayers, Compression and others. "
                       "Default: ['Tin', 'Tout']")

    # --- Log Parameters Configuration ---
    parser.add_argument("--log_params", nargs='+', default=['num_rounds', 'local_epochs',"Tin",'seed'],
                       type=str, help="Hyperparameters to include in log filename. "
                       "Available options: num_rounds, local_epochs, seed, baselr, d_model, nhead, nlayers, compression, Tin, Tout and others. "
                       "Default: ['num_rounds', 'local_epochs']")

    parser.add_argument("--info", required=False, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    federated_train(args)
