import torch
import numpy as np
import os
import json
import yaml
from collections import OrderedDict
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng
from lib.utils.logger import get_color_logger as get_logger
from lib.utils.results_logger import log_results_to_jsonl
from lib.models.Server import Server
from lib.models.Client import Client
from lib.utils.privacy import add_noise_to_memory
from lib.utils.naming import build_log_filename, build_checkpoint_dirname
from lib.utils.load_args import load_additional_config, merge_args

# #############################################################################
# Main FeDPM Orchestration
# #############################################################################

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