import torch
import torch.nn.functional as F

import os
import json
from collections import OrderedDict
from lib.models.Encoder_and_Retrieval import PMR # Import PMR (Prototypical Memories Retrieval)

# #############################################################################
# 2. FeDPM Components: Server and Client
# #############################################################################

class Server:
    """
    The Server in a FeDPM setup.
    It holds the global model and aggregates updates from clients.
    """
    def __init__(self, args, device, logger, aggregation_strategy='cos_similarity', similarity_threshold=0.5,
                 memory_fill_strategy='client_personalized', gamma=0.95):
        self.logger = logger
        self.args = args
        self.logger.info(f"Initializing Server with aggregation strategy: {aggregation_strategy}")
        self.device = device
        self.strategy = aggregation_strategy
        self.similarity_threshold = similarity_threshold
        self.memory_fill_strategy = memory_fill_strategy
        self.gamma = gamma
        
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
        
        # TODO: You can change to matrix operations for efficiency if needed.
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
