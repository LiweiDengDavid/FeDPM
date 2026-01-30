import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from lib.utils.results_logger import human_count_params
# from lib.models import get_model_class
from lib.models.decode import XcodeYtimeDecoder, MuStdModel
from lib.models.revin import RevIN
from lib.models.metrics import pearsoncor
from lib.utils.data_utils import create_time_series_dataloader, get_params, loss_fn
from lib.models.Encoder_and_Retrieval import PMR, Encoder # Import Encoder and PMR (Prototypical Memories Retrieval)

# #############################################################################
# 2. FeDPM Components: Server and Client
# #############################################################################

class Server:
    """
    The Server in a FeDPM setup.
    It holds the global model and aggregates updates from clients.
    """
    def __init__(self, args, device, logger, aggregation_strategy='fedavg', similarity_threshold=0.5,
                 memory_fill_strategy='random_isolated', gamma=0.95):
        self.logger = logger
        self.args = args
        self.logger.info(f"Initializing Server with aggregation strategy: {aggregation_strategy}")
        self.device = device
        self.strategy = aggregation_strategy
        self.similarity_threshold = similarity_threshold
        self.memory_fill_strategy = memory_fill_strategy
        self.gamma = gamma
        
        # The global model only contains the vector quantizer (memory)
        # PMR: Prototypical Memories Retrieval
        self.global_model = PMR(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost
        ).to(self.device)

    def get_global_weights(self):
        """Returns the state dictionary of the global model."""
        return self.global_model.state_dict()

    def _fed_avg(self, client_updates, client_data_sizes):
        """ Aggregates using weighted Federated Averaging for memories. """
        self.logger.info("Server: Aggregating client memories using FedAvg (weighted)...")
        total_data_size = sum(client_data_sizes)
        agg_weights = OrderedDict()

        # client_updates are memory tensors
        for i, memory_tensor in enumerate(client_updates):
            data_size = client_data_sizes[i]
            weight_factor = data_size / total_data_size
            if i == 0:
                agg_weights['_embedding.weight'] = memory_tensor.clone() * weight_factor
            else:
                agg_weights['_embedding.weight'] += memory_tensor.clone() * weight_factor
        
        return agg_weights

    def _aggregate_memory_by_cos_similarity(self, client_memories, client_data_sizes=None, client_usage_counts=None):
        """
        Aggregates memories based on cosine similarity of individual prototypes.
        
        Args:
            client_memories: List of memory tensors
            client_data_sizes: List of client data sizes (for client_personalized strategy)
            client_usage_counts: List of usage count tensors (for client_personalized strategy)
        """
        self.logger.info(f"Server: Aggregating VQ memory using cosine similarity strategy (Similarity threshold {self.similarity_threshold})...")
        num_clients = len(client_memories)
        num_prototypes, prototype_dim = client_memories[0].shape
        
        # TODO: You can change to matrix operations (The code of the next part, but it has been commented out) for efficiency if needed.
        # 1. Build adjacency list for the graph of all prototypes
        # A node is represented by a tuple (client_id, prototype_index)
        # represent client c prototype i as node (c, i), connected to others if similar
        adj = {(c, i): [] for c in range(num_clients) for i in range(num_prototypes)} 
        
        for c1 in range(num_clients):
            for c2 in range(c1 + 1, num_clients):
                cb1 = client_memories[c1]
                cb2 = client_memories[c2]
                
                # Calculate cosine similarity matrix between two memories
                sim_matrix = F.cosine_similarity(cb1.unsqueeze(1), cb2.unsqueeze(0), dim=-1)
                
                # Add edges for similarities > threshold
                similar_indices = torch.where(sim_matrix > self.similarity_threshold)
                for i, j in zip(*similar_indices):
                    node1 = (c1, i.item())
                    node2 = (c2, j.item())
                    adj[node1].append(node2)
                    adj[node2].append(node1)

        # 2. Find connected components (clusters of similar prototypes) using BFS
        visited = set()
        clusters = []
        for c in range(num_clients):
            for i in range(num_prototypes):
                node = (c, i)
                if node not in visited:
                    cluster = []
                    q = [node]
                    visited.add(node)
                    while q:
                        curr_node = q.pop(0)
                        cluster.append(curr_node)
                        for neighbor in adj[curr_node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    clusters.append(cluster)
        
        self.logger.info(f"Found {len(clusters)} clusters among {num_clients * num_prototypes} prototypes.")

        # 3. Aggregate prototypes within each cluster
        # Store as (mean_prototype, cluster_size) tuples
        # Only consider clusters with 2 or more nodes as valid clusters
        aggregated_prototypes_info = []
        valid_clusters = []
        isolated_nodes = set()
        
        for cluster in clusters:
            if len(cluster) >= 2:
                # Valid cluster: has 2 or more nodes
                prototypes_in_cluster = [client_memories[c_id][v_idx] for c_id, v_idx in cluster]
                # Simple arithmetic mean for prototypes in a cluster
                mean_prototype = torch.stack(prototypes_in_cluster).mean(dim=0)
                aggregated_prototypes_info.append((mean_prototype, len(cluster)))
                valid_clusters.append(cluster)
            else:
                # Isolated node: cluster with only 1 node
                isolated_nodes.update(cluster)
        
        self.logger.info(f"Valid clusters (size >= 2): {len(valid_clusters)}, Isolated nodes: {len(isolated_nodes)}")

        # Sort aggregated prototypes by cluster size (descending)
        # This prioritizes prototypes that represent larger clusters (more common patterns)
        aggregated_prototypes_info.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the prototypes
        aggregated_prototypes = [x[0] for x in aggregated_prototypes_info]

        # 4. Construct the new global memory
        new_memory = torch.zeros_like(client_memories[0])
        
        # Apply gamma to limit shared prototypes
        max_shared = int(num_prototypes * self.gamma)
        num_available_aggregated = len(aggregated_prototypes)
        
        # Number of shared prototypes to actually use (limited by gamma and availability)
        num_aggregated = min(num_available_aggregated, max_shared)
        
        if num_aggregated > 0:
            new_memory[:num_aggregated] = torch.stack(aggregated_prototypes[:num_aggregated])
            
        # Fill the rest based on strategy
        remaining_slots = num_prototypes - num_aggregated
        if remaining_slots > 0:
                if self.memory_fill_strategy == 'random_isolated':
                    # Collect isolated prototypes (nodes not in valid clusters)
                    isolated_prototypes = []
                    for c_id, v_idx in isolated_nodes:
                        isolated_prototypes.append(client_memories[c_id][v_idx])
                    
                    if len(isolated_prototypes) > 0:
                        # Randomly select from isolated prototypes to fill the rest
                        num_to_fill = min(remaining_slots, len(isolated_prototypes))
                        indices = torch.randperm(len(isolated_prototypes))[:num_to_fill]
                        for i, idx in enumerate(indices):
                            new_memory[num_aggregated + i] = isolated_prototypes[idx]
                        self.logger.info(f"Filled {num_to_fill} slots with isolated prototypes (from {len(isolated_prototypes)} available).")
                
                elif self.memory_fill_strategy == 'client_personalized':
                    # Client-personalized strategy: each client gets its own personalized prototypes
                    # Returns a list of memories (one per client) instead of a single global memory
                    self.logger.info(f"Using client_personalized strategy: each client will have personalized prototypes in remaining {remaining_slots} slots.")
                    
                    # Build personalized memories for each client
                    personalized_memories = []
                    
                    for c_id in range(num_clients):
                        # Start with the shared aggregated prototypes
                        client_memory = new_memory.clone()
                        
                        # Get this client's isolated nodes
                        client_isolated_nodes = [(cid, vid) for cid, vid in isolated_nodes if cid == c_id]
                        
                        if len(client_isolated_nodes) > 0:
                            # Calculate cross-client similarity and usage frequency for each isolated prototype
                            # We want prototypes with LOWEST similarity to other clients (most personalized)
                            # AND high usage frequency (not noise)
                            prototype_similarities = []
                            
                            for node in client_isolated_nodes:
                                c_id_node, v_idx_node = node
                                prototype = client_memories[c_id_node][v_idx_node]
                                
                                # Calculate average similarity to all OTHER clients' memories
                                total_sim = 0.0
                                count = 0
                                for other_c_id in range(num_clients):
                                    if other_c_id != c_id:
                                        # Normalize prototypes for cosine similarity
                                        normalized_prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
                                        normalized_other_memory = F.normalize(client_memories[other_c_id], p=2, dim=1)
                                        
                                        # Compute similarity with all prototypes in other client's memory
                                        similarities = torch.mm(normalized_prototype, normalized_other_memory.t())
                                        mean_sim = similarities.mean().item()
                                        total_sim += mean_sim
                                        count += 1
                                
                                avg_cross_client_sim = total_sim / count if count > 0 else 0.0
                                
                                # Get usage frequency for this prototype
                                if client_usage_counts is not None and c_id < len(client_usage_counts):
                                    usage_count = client_usage_counts[c_id][v_idx_node].item()
                                else:
                                    usage_count = 1.0  # Default if usage not tracked
                                
                                # Store (node, prototype, avg_similarity, usage_count)
                                prototype_similarities.append((node, prototype, avg_cross_client_sim, usage_count))
                            
                            # Normalize usage counts within this client's isolated prototypes
                            max_usage = max(item[3] for item in prototype_similarities)
                            
                            # Compute personalization score for each prototype
                            scored_prototypes = []
                            for node, prototype, avg_sim, usage in prototype_similarities:
                                # Normalize usage to [0, 1]
                                normalized_usage = usage / max_usage if max_usage > 0 else 0.0
                                
                                # Personalization score = -similarity + lambda * normalized_usage
                                # Lower cross-client similarity + higher usage = higher score
                                lambda_weight = 1.0  # Balance between similarity and usage (adjustable)
                                personalization_score = -avg_sim + lambda_weight * normalized_usage
                                
                                scored_prototypes.append((node, prototype, personalization_score, avg_sim, usage))
                            
                            # Sort by personalization score (descending): highest score = most personalized + most used
                            scored_prototypes.sort(key=lambda x: x[2], reverse=True)
                            
                            # Select top remaining_slots most personalized prototypes
                            num_to_select = min(remaining_slots, len(scored_prototypes))
                            selected_prototypes = [item[1] for item in scored_prototypes[:num_to_select]]
                            
                            # Fill the remaining slots with selected personalized prototypes
                            for i, vec in enumerate(selected_prototypes):
                                client_memory[num_aggregated + i] = vec
                            
                            # Log selection details
                            if num_to_select > 0:
                                top_score = scored_prototypes[0]
                                bottom_score = scored_prototypes[num_to_select - 1]
                                self.logger.info(
                                    f"Client {c_id}: Filled {num_to_select} personalized slots | "
                                    f"Top: score={top_score[2]:.3f} (sim={top_score[3]:.3f}, usage={top_score[4]:.0f}) | "
                                    f"Bottom: score={bottom_score[2]:.3f} (sim={bottom_score[3]:.3f}, usage={bottom_score[4]:.0f})"
                                )
                        else:
                            self.logger.warning(f"Client {c_id}: No isolated prototypes available for personalization.")
                        
                        personalized_memories.append(client_memory)
                    
                    # Return the list of personalized memories instead of a single global one
                    return personalized_memories
                
                elif self.memory_fill_strategy == 'none':
                    # Do nothing, remaining slots stay as zeros (from torch.zeros_like initialization)
                    pass
                else:
                    self.logger.warning(f"Unknown memory fill strategy: {self.memory_fill_strategy}. Using 'none'.")
        
        return new_memory
    
    def _cos_similarity(self, client_updates, client_data_sizes, client_usage_counts=None):
        """
        Aggregation via Cosine Similarity for VQ memory.
        client_updates are expected to be a list of memory tensors.
        
        Args:
            client_updates: List of memory tensors
            client_data_sizes: List of client data sizes
            client_usage_counts: List of usage count tensors (for personalization)
        
        Returns:
            - For 'random_isolated' or 'none' strategies: OrderedDict with single global memory
            - For 'client_personalized' strategy: List of OrderedDicts (one per client)
        """
        self.logger.info(f"Server: Aggregating using 'cos_similarity' strategy for memories (Similarity threshold {self.similarity_threshold})...")
        
        # --- Aggregate VQ Memory using Cosine Similarity ---
        result = self._aggregate_memory_by_cos_similarity(
            client_updates, 
            client_data_sizes=client_data_sizes,
            client_usage_counts=client_usage_counts
        )
        
        # Check if result is a list (client_personalized) or a single tensor
        if isinstance(result, list):
            # client_personalized strategy: return list of state_dicts
            final_agg_weights = []
            for client_memory in result:
                client_state_dict = OrderedDict()
                client_state_dict['_embedding.weight'] = client_memory
                final_agg_weights.append(client_state_dict)
            return final_agg_weights
        else:
            # Other strategies: return single state_dict
            final_agg_weights = OrderedDict()
            final_agg_weights['_embedding.weight'] = result
            return final_agg_weights

    def aggregate(self, client_updates, client_data_sizes, client_usage_counts=None):
        """
        Aggregates client model updates based on the chosen strategy.
        
        Args:
            client_updates: List of client model updates
            client_data_sizes: List of client data sizes
            client_usage_counts: List of usage count tensors (for personalization)
            
        Note:
            For client_personalized strategy, the global_model will store the shared part,
            and personalized parts are returned separately for each client.
        """
        if self.strategy == 'fedavg':
            agg_weights = self._fed_avg(client_updates, client_data_sizes)
            self.global_model.load_state_dict(agg_weights)
            self.logger.info(f"Server: Aggregation complete by '{self.strategy}' strategy. Global model updated.")
            return None  # No personalized weights
            
        elif self.strategy == 'cos_similarity':
            agg_weights = self._cos_similarity(
                client_updates, 
                client_data_sizes, 
                client_usage_counts=client_usage_counts
            )
            
            # Check if we have personalized memories
            if isinstance(agg_weights, list):
                # client_personalized: load the first client's memory to global model (they share the same aggregated part)
                # but return the list so clients can get their personalized versions
                self.global_model.load_state_dict(agg_weights[0])
                self.logger.info(f"Server: Aggregation complete by '{self.strategy}' strategy with 'client_personalized' fill. "
                               f"Each client will receive personalized memory.")
                return agg_weights  # Return list of personalized memories
            else:
                # Standard aggregation
                self.global_model.load_state_dict(agg_weights)
                self.logger.info(f"Server: Aggregation complete by '{self.strategy}' strategy. Global model updated.")
                return None  # No personalized weights
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

class Client:
    """
    A Client in a FeDPM setup.
    It has its own local data and local models (decode/mustd).
    It receives the global model, trains on its data, and sends back the update.
    """
    def __init__(self, client_id, args, device, logger):
        self.client_id = client_id
        self.args = args
        self.device = device
        self.data_type = args.client_data_types[client_id]
        self.logger = logger
        self.logger.info(f"Initializing Client {self.client_id} with dataset: {self.data_type}")

        # Load client-specific data
        params = get_params(self.data_type)
        params["dataroot"]= os.path.join(args.base_data_path+self.data_type,f"Tin{args.Tin}_Tout{args.Tout}")
        self.dataloaders = create_time_series_dataloader(params["dataroot"], params["batchsize"],logger=self.logger)
        
        # Apply few-shot sampling if ratio < 1.0
        few_shot_ratio = getattr(args, 'few_shot_ratio', 1.0)
        if few_shot_ratio < 1.0 and 'train' in self.dataloaders:
            original_train_dataset = self.dataloaders['train'].dataset
            total_samples = len(original_train_dataset)
            num_few_shot_samples = int(total_samples * few_shot_ratio)
            
            # Take the first num_few_shot_samples for deterministic behavior
            few_shot_indices = list(range(num_few_shot_samples))
            few_shot_dataset = Subset(original_train_dataset, few_shot_indices)
            
            # Create a new DataLoader with the subset
            self.dataloaders['train'] = DataLoader(
                few_shot_dataset,
                batch_size=self.dataloaders['train'].batch_size,
                shuffle=True,
                num_workers=0 
            )
            
            self.logger.info(f"Client {self.client_id}: Few-shot mode enabled. "
                           f"Using {num_few_shot_samples}/{total_samples} samples ({few_shot_ratio*100:.1f}%)")
        
        self.data_size = len(self.dataloaders['train'].dataset) if 'train' in self.dataloaders else 0

        # Initialize local models (specific to each client)
        Sin, Sout = params["Sin"], params["Sout"]
        dim = args.embedding_dim if not args.onehot else args.num_embeddings
        
        # Get encoder and decoder configuration from args (loaded from Ablation_args.yaml)
        encoder_type = getattr(args, 'encoder_config_encoder_type', 'cnn')
        decoder_type = getattr(args, 'decoder_config_decoder_type', 'transformer')
        transformer_nhead = getattr(args, 'encoder_config_transformer_nhead', 4)
        transformer_layers = getattr(args, 'encoder_config_transformer_layers', 2)
        encoder_dropout = getattr(args, 'encoder_config_encoder_dropout', 0.1)
        rnn_layers = getattr(args, 'encoder_config_rnn_layers', 2)
        
        # Get VQVAE compression_factor
        self.compression_factor = args.compression_factor

        # Encoder is now a local model
        self.model_encoder = Encoder(
            in_channels=1,
            num_hiddens=args.block_hidden_size,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.res_hidden_size,
            embedding_dim=args.embedding_dim,
            compression_factor=args.compression_factor,
            encoder_type=encoder_type,
            seq_len=self.args.Tin,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            dropout=encoder_dropout,
            rnn_layers=rnn_layers
        ).to(self.device)

        # VQ is also local, but its memory will be synced with the server
        # PMR: Prototypical Memories Retrieval
        self.model_vq = PMR(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost
        ).to(self.device)

        # Initialize memory usage frequency counter for personalization
        self.memory_usage_count = torch.zeros(
            args.num_embeddings, 
            device=self.device
        )

        self.model_decode = XcodeYtimeDecoder(
            d_in=dim,
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            seq_in_len=args.Tin // args.compression_factor,
            seq_out_len=args.Tout,
            dropout=0.0,
            decoder_type=decoder_type,
            compression_factor=args.compression_factor,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.res_hidden_size,
        ).to(self.device)


        # Determine if MuStdModel should be disabled for this client
        # It is disabled if this client is in the specific list
        specific_disable_list = getattr(args, 'clients_disable_mustd', [])
        should_disable = (self.data_type in specific_disable_list)

        if should_disable:
            self.model_mustd = None
            self.revin_in = RevIN(num_features=Sin, affine=False, device=self.device)
            self.revin_out = RevIN(num_features=Sout, affine=False, device=self.device)
            self.logger.info(f"Client {self.client_id} ({self.data_type}): MuStdModel disabled. Using standard RevIN.")
        else:
            self.model_mustd = MuStdModel(
                Tin=args.Tin, Tout=args.Tout, hidden_dims=[512, 512], dropout=0.2, is_mlp=True
            ).to(self.device)
            self.model_mustd.revin_in = RevIN(num_features=Sin, affine=False, device=self.device)
            self.model_mustd.revin_out = RevIN(num_features=Sout, affine=False, device=self.device)
            self.logger.info(f"Client {self.client_id} ({self.data_type}): MuStdModel enabled.")

    def get_local_models(self):
        """Returns the local models of the client for checkpointing."""
        models = {
            f'client_{self.client_id}_{self.data_type}_encoder': self.model_encoder,
            f'client_{self.client_id}_{self.data_type}_decode': self.model_decode,
        }
        if self.model_mustd is not None:
            models[f'client_{self.client_id}_{self.data_type}_mustd'] = self.model_mustd
        return models

    def local_train(self, global_memory_weights):
        """
        Performs local training on the client's data.
        """
        if not self.dataloaders or 'train' not in self.dataloaders:
            self.logger.warning(f"Client {self.client_id}: No training data available!")
            return None

        self.logger.info(f"Client {self.client_id}: Starting local training for {self.args.local_epochs} epochs.")
        
        # Load the global memory into the local VQ model
        self.model_vq.load_state_dict(global_memory_weights)

        # Set models to train mode
        self.model_encoder.train()
        self.model_vq.train()
        self.model_decode.train()
        if self.model_mustd is not None:
            self.model_mustd.train()

        # Optimizer for all local parameters
        all_params = (
            list(self.model_encoder.parameters()) +
            list(self.model_vq.parameters()) +
            list(self.model_decode.parameters())
        )
        if self.model_mustd is not None:
            all_params += list(self.model_mustd.parameters())
        
        param_str, total_params = human_count_params(all_params)
        self.logger.info(f"Client {self.client_id}: Total local model parameters: {param_str} ({total_params})")

        optimizer = torch.optim.Adam(all_params, lr=self.args.baselr)
        
        lossfn_instance = loss_fn(self.args.loss_type, beta=self.args.beta)

        # Reset memory usage counter before training
        self.memory_usage_count.zero_()

        for epoch in range(self.args.local_epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(self.dataloaders['train']):
                x, y = x.to(self.device), y.to(self.device)
                bs, _, Sout = y.shape
                
                # Forward pass through the combined model
                # 1. Get codes from the local VQ-VAE model
                if self.model_mustd is not None:
                    norm_x = self.model_mustd.revin_in(x, "norm")
                    norm_y = self.model_mustd.revin_out(y, "norm")
                else:
                    norm_x = self.revin_in(x, "norm")
                    norm_y = self.revin_out(y, "norm")

                norm_x_permuted = torch.permute(norm_x, (0, 2, 1))
                norm_x_reshaped = norm_x_permuted.reshape(-1, norm_x_permuted.shape[-1])
                
                latent_x = self.model_encoder(norm_x_reshaped, self.compression_factor)
                vq_loss, quantized_x, perplexity, encodings, encoding_indices, _ = self.model_vq(latent_x)
                
                # Track memory usage frequency (vectorized for efficiency)
                used_indices = encoding_indices.flatten()
                # Use scatter_add for fast batched updates
                self.memory_usage_count.scatter_add_(0, used_indices, torch.ones_like(used_indices, dtype=self.memory_usage_count.dtype))
                
                # 2. Use codes to predict with the local decode model
                xcodes = quantized_x.reshape(bs, x.shape[-1], self.args.embedding_dim, -1)
                xcodes = torch.permute(xcodes, (0, 1, 3, 2)) # B, S, TC, D
                xcodes = xcodes.reshape(bs * x.shape[-1], xcodes.shape[2], xcodes.shape[3]) # B*S, TC, D
                xcodes = torch.permute(xcodes, (1, 0, 2)) # TC, B*S, D

                ytime = self.model_decode(xcodes)
                ytime = ytime.reshape((bs, Sout, self.args.Tout)).permute(0, 2, 1)

                if self.model_mustd is not None:
                    # 3. Get mu/std from local mustd model
                    times = torch.permute(x, (0, 2, 1)).reshape((-1, x.shape[1]))
                    ymeanstd = self.model_mustd(times).reshape((bs, Sout, 2)).permute(0, 2, 1)
                    ymean = ymeanstd[:, 0, :].unsqueeze(1)
                    ystd = ymeanstd[:, 1, :].unsqueeze(1)

                    # 4. Calculate loss
                    loss_decode = lossfn_instance(ytime, norm_y.reshape(bs, self.args.Tout, Sout))
                    loss_mu = lossfn_instance(self.model_mustd.revin_out.mean - self.model_mustd.revin_in.mean, ymean)
                    loss_std = lossfn_instance(self.model_mustd.revin_out.stdev - self.model_mustd.revin_in.stdev, ystd)
                    loss_all = lossfn_instance(ytime * (ystd.detach() + self.model_mustd.revin_in.stdev) 
                                            + (ymean.detach() + self.model_mustd.revin_in.mean),y)

                    # Total loss includes VQ-VAE loss
                    loss = loss_decode + loss_mu + loss_std + loss_all + vq_loss
                else:
                    # 4. Calculate loss (without MuStd)
                    loss_decode = lossfn_instance(ytime, norm_y.reshape(bs, self.args.Tout, Sout))
                    
                    # Denormalize using input statistics (revin_in)
                    # y_pred = ytime * std_in + mean_in
                    y_pred = ytime * self.revin_in.stdev + self.revin_in.mean
                    loss_all = lossfn_instance(y_pred, y)
                    
                    loss = loss_decode + loss_all + vq_loss

                # Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(self.dataloaders['train'])
            self.logger.info(f"Client {self.client_id} | Epoch {epoch+1}/{self.args.local_epochs} | Avg Loss: {avg_epoch_loss:.4f}")

        # Log memory usage statistics
        usage_stats = {
            'min': self.memory_usage_count.min().item(),
            'max': self.memory_usage_count.max().item(),
            'mean': self.memory_usage_count.mean().item(),
            'std': self.memory_usage_count.std().item(),
            'num_unused': (self.memory_usage_count == 0).sum().item()
        }
        self.logger.info(f"Client {self.client_id} | Memory usage stats: "
                        f"min={usage_stats['min']:.0f}, max={usage_stats['max']:.0f}, "
                        f"mean={usage_stats['mean']:.1f}, std={usage_stats['std']:.1f}, "
                        f"unused_prototypes={usage_stats['num_unused']}/{len(self.memory_usage_count)}")

        # Return only the updated memory tensor
        return self.model_vq._embedding.weight.detach()

    def evaluate(self, global_memory_weights, split='val', use_local_memory=False):
        """
        Evaluates the current global model on the client's local data.
        
        Args:
            global_memory_weights: Global memory weights from server (can be None if use_local_memory=True)
            split: 'val' for validation, 'test' for final testing
            use_local_memory: If True, use the client's own local memory instead of global
        """
        if split not in self.dataloaders:
            self.logger.warning(f"Client {self.client_id}: No '{split}' dataloader available.")
            return None

        self.logger.info(f"Client {self.client_id}: Evaluating on '{split}' data{' (using local memory)' if use_local_memory else ''}.")

        # Load the global memory into the local VQ model (unless using local memory)
        if not use_local_memory and global_memory_weights is not None:
            self.model_vq.load_state_dict(global_memory_weights)
        elif use_local_memory:
            self.logger.info(f"Client {self.client_id}: Using local memory for evaluation (no aggregation).")
        else:
            self.logger.warning(f"Client {self.client_id}: No global memory provided and use_local_memory=False.")

        # Set all models to evaluation mode
        self.model_encoder.eval()
        self.model_vq.eval()
        self.model_decode.eval()
        if self.model_mustd is not None:
            self.model_mustd.eval()

        total_loss = 0
        total_mse = 0
        total_mae = 0
        total_corr = 0
        total_corr_count = 0  # Add a counter for the number of correlation values
        total_samples = 0
        lossfn_instance = loss_fn(self.args.loss_type, beta=self.args.beta)

        with torch.no_grad():
            for x, y in self.dataloaders[split]:
                x, y = x.to(self.device), y.to(self.device)
                bs, _, Sout = y.shape
                
                # --- Forward pass (CORRECTED: Do NOT use y statistics for denormalization) ---
                if self.model_mustd is not None:
                    # Normalize input x using revin_in
                    norm_x = self.model_mustd.revin_in(x, "norm")
                    # For loss calculation only, normalize y (but DON'T use these stats for denorm!)
                    # We need to save revin_out stats before calling norm on y
                    saved_revin_out_mean = self.model_mustd.revin_out.mean.clone() if hasattr(self.model_mustd.revin_out, 'mean') else None
                    saved_revin_out_stdev = self.model_mustd.revin_out.stdev.clone() if hasattr(self.model_mustd.revin_out, 'stdev') else None
                    norm_y = self.model_mustd.revin_out(y, "norm")
                else:
                    norm_x = self.revin_in(x, "norm")
                    # For loss only
                    saved_revin_out_mean = self.revin_out.mean.clone() if hasattr(self.revin_out, 'mean') else None
                    saved_revin_out_stdev = self.revin_out.stdev.clone() if hasattr(self.revin_out, 'stdev') else None
                    norm_y = self.revin_out(y, "norm")

                norm_x_permuted = torch.permute(norm_x, (0, 2, 1))
                norm_x_reshaped = norm_x_permuted.reshape(-1, norm_x_permuted.shape[-1])
                latent_x = self.model_encoder(norm_x_reshaped, self.compression_factor)
                vq_loss, quantized_x, _, _, _, _ = self.model_vq(latent_x)
                xcodes = quantized_x.reshape(bs, x.shape[-1], self.args.embedding_dim, -1)
                xcodes = torch.permute(xcodes, (0, 1, 3, 2))
                xcodes = xcodes.reshape(bs * x.shape[-1], xcodes.shape[2], xcodes.shape[3])
                xcodes = torch.permute(xcodes, (1, 0, 2))
                
                ytime = self.model_decode(xcodes)
                ytime = ytime.reshape((bs, Sout, self.args.Tout)).permute(0, 2, 1)
                
                if self.model_mustd is not None:
                    times = torch.permute (x, (0, 2, 1)).reshape((-1, x.shape[1]))
                    ymeanstd = self.model_mustd(times).reshape((bs, Sout, 2)).permute(0, 2, 1)
                    ymean = ymeanstd[:, 0, :].unsqueeze(1)
                    ystd = ymeanstd[:, 1, :].unsqueeze(1)
                    
                    # --- Calculate loss (using normalized y) ---
                    loss_decode = lossfn_instance(ytime, norm_y.reshape(bs, self.args.Tout, Sout))
                    
                    # For mu/std loss, we need to use the y statistics (for training consistency)
                    # But restore them after calculation to avoid contamination
                    if saved_revin_out_mean is not None:
                        loss_mu = lossfn_instance(self.model_mustd.revin_out.mean - self.model_mustd.revin_in.mean, ymean)
                        loss_std = lossfn_instance(self.model_mustd.revin_out.stdev - self.model_mustd.revin_in.stdev, ystd)
                    else:
                        loss_mu = loss_std = torch.tensor(0.0, device=self.device)
                    
                    loss = loss_decode + loss_mu + loss_std + vq_loss
                    
                    #  Denormalize using INPUT statistics + predicted mu/std
                    # This matches the inference logic in central_main.py (scheme=1)
                    pred_denorm = ytime * (ystd + self.model_mustd.revin_in.stdev) + (ymean + self.model_mustd.revin_in.mean)
                else:
                    # --- Calculate loss and metrics ---
                    loss_decode = lossfn_instance(ytime, norm_y.reshape(bs, self.args.Tout, Sout))
                    loss = loss_decode + vq_loss
                    
                    #  Denormalize using input statistics 
                    pred_denorm = ytime * self.revin_in.stdev + self.revin_in.mean

                total_loss += loss.item() * bs
                total_mse += F.mse_loss(pred_denorm, y).item() * bs
                total_mae += F.l1_loss(pred_denorm, y).item() * bs
                total_corr += pearsoncor(pred_denorm, y, reduction="sum").item()
                total_corr_count += bs * Sout  # Accumulate the correct number of values
                total_samples += bs

        avg_loss = total_loss / total_samples
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_corr = total_corr / total_corr_count if total_corr_count > 0 else 0

        return {'loss': avg_loss, 'mse': avg_mse, 'mae': avg_mae, 'corr': avg_corr}
