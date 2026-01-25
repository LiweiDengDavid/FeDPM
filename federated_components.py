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
from lib.models import get_model_class
from lib.models.decode import XcodeYtimeDecoder, MuStdModel
from lib.models.revin import RevIN
from lib.models.metrics import pearsoncor
from lib.utils.data_utils import create_time_series_dataloader, get_params, loss_fn
from lib.models.vqvae import VectorQuantizer, Encoder # Import Encoder and VectorQuantizer

# #############################################################################
# 2. Federated Learning Components: Server and Client
# #############################################################################

class Server:
    """
    The Server in a Federated Learning setup.
    It holds the global model and aggregates updates from clients.
    """
    def __init__(self, vqvae_config, device, logger, aggregation_strategy='fedavg', similarity_threshold=0.5,
                 enable_diversity_loss=False, diversity_loss_weight=0.1, diversity_loss_type='repulsion', 
                 diversity_temperature=0.5, codebook_fill_strategy='random_isolated', gamma=0.95):
        self.logger = logger
        self.logger.info(f"Initializing Server with aggregation strategy: {aggregation_strategy}")
        self.device = device
        self.strategy = aggregation_strategy
        self.similarity_threshold = similarity_threshold
        self.codebook_fill_strategy = codebook_fill_strategy
        self.gamma = gamma
        
        # Diversity loss parameters
        self.enable_diversity_loss = enable_diversity_loss
        self.diversity_loss_weight = diversity_loss_weight
        self.diversity_loss_type = diversity_loss_type
        self.diversity_temperature = diversity_temperature
        
        if self.enable_diversity_loss:
            self.logger.info(f"Codebook diversity loss enabled:")
            self.logger.info(f"  Type: {self.diversity_loss_type}")
            self.logger.info(f"  Weight: {self.diversity_loss_weight}")
            self.logger.info(f"  Temperature: {self.diversity_temperature}")
        
        # The global model only contains the vector quantizer (codebook)
        self.global_model = VectorQuantizer(
            num_embeddings=vqvae_config['num_embeddings'],
            embedding_dim=vqvae_config['embedding_dim'],
            commitment_cost=vqvae_config['commitment_cost']
        ).to(self.device)

    def get_global_weights(self):
        """Returns the state dictionary of the global model."""
        return self.global_model.state_dict()

    def _fed_avg(self, client_updates, client_data_sizes):
        """ Aggregates using weighted Federated Averaging for codebooks. """
        self.logger.info("Server: Aggregating client codebooks using FedAvg (weighted)...")
        total_data_size = sum(client_data_sizes)
        agg_weights = OrderedDict()

        # client_updates are codebook tensors
        for i, codebook_tensor in enumerate(client_updates):
            data_size = client_data_sizes[i]
            weight_factor = data_size / total_data_size
            if i == 0:
                agg_weights['_embedding.weight'] = codebook_tensor.clone() * weight_factor
            else:
                agg_weights['_embedding.weight'] += codebook_tensor.clone() * weight_factor
        
        return agg_weights

    def _aggregate_codebook_by_cos_similarity(self, client_codebooks, round_num=None, save_dir=None, client_names=None, client_data_sizes=None, client_usage_counts=None):
        """
        Aggregates codebooks based on cosine similarity of individual code vectors.
        
        Args:
            client_codebooks: List of codebook tensors
            round_num: Current round number (for visualization)
            save_dir: Directory to save visualizations (if None, no visualization)
            client_names: List of client names (optional)
            client_data_sizes: List of client data sizes (for client_personalized strategy)
            client_usage_counts: List of usage count tensors (for client_personalized strategy)
        """
        self.logger.info(f"Server: Aggregating VQ codebook using cosine similarity strategy (Similarity threshold {self.similarity_threshold})...")
        num_clients = len(client_codebooks)
        num_codes, code_dim = client_codebooks[0].shape
        
        # TODO: You can change to matrix operations (The code of the next part, but it has been commented out) for efficiency if needed.
        # 1. Build adjacency list for the graph of all code vectors
        # A node is represented by a tuple (client_id, vector_index)
        # represent client c code vector i as node (c, i), connected to others if similar
        adj = {(c, i): [] for c in range(num_clients) for i in range(num_codes)} 
        
        for c1 in range(num_clients):
            for c2 in range(c1 + 1, num_clients):
                cb1 = client_codebooks[c1]
                cb2 = client_codebooks[c2]
                
                # Calculate cosine similarity matrix between two codebooks
                sim_matrix = F.cosine_similarity(cb1.unsqueeze(1), cb2.unsqueeze(0), dim=-1)
                
                # Add edges for similarities > threshold
                similar_indices = torch.where(sim_matrix > self.similarity_threshold)
                for i, j in zip(*similar_indices):
                    node1 = (c1, i.item())
                    node2 = (c2, j.item())
                    adj[node1].append(node2)
                    adj[node2].append(node1)

        # 2. Find connected components (clusters of similar vectors) using BFS
        visited = set()
        clusters = []
        for c in range(num_clients):
            for i in range(num_codes):
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
        
        self.logger.info(f"Found {len(clusters)} clusters among {num_clients * num_codes} code vectors.")

        # 3. Aggregate vectors within each cluster
        # Store as (mean_vector, cluster_size) tuples
        # Only consider clusters with 2 or more nodes as valid clusters
        aggregated_vectors_info = []
        valid_clusters = []
        isolated_nodes = set()
        
        for cluster in clusters:
            if len(cluster) >= 2:
                # Valid cluster: has 2 or more nodes
                vectors_in_cluster = [client_codebooks[c_id][v_idx] for c_id, v_idx in cluster]
                # Simple arithmetic mean for vectors in a cluster
                mean_vector = torch.stack(vectors_in_cluster).mean(dim=0)
                aggregated_vectors_info.append((mean_vector, len(cluster)))
                valid_clusters.append(cluster)
            else:
                # Isolated node: cluster with only 1 node
                isolated_nodes.update(cluster)
        
        self.logger.info(f"Valid clusters (size >= 2): {len(valid_clusters)}, Isolated nodes: {len(isolated_nodes)}")

        # Sort aggregated vectors by cluster size (descending)
        # This prioritizes vectors that represent larger clusters (more common patterns)
        aggregated_vectors_info.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the vectors
        aggregated_vectors = [x[0] for x in aggregated_vectors_info]

        # 4. Construct the new global codebook
        new_codebook = torch.zeros_like(client_codebooks[0])
        
        # Apply gamma to limit shared vectors
        max_shared = int(num_codes * self.gamma)
        num_available_aggregated = len(aggregated_vectors)
        
        # Number of shared vectors to actually use (limited by gamma and availability)
        num_aggregated = min(num_available_aggregated, max_shared)
        
        if num_aggregated > 0:
            new_codebook[:num_aggregated] = torch.stack(aggregated_vectors[:num_aggregated])
            
        # Fill the rest based on strategy
        remaining_slots = num_codes - num_aggregated
        if remaining_slots > 0:
                if self.codebook_fill_strategy == 'random_isolated':
                    # Collect isolated vectors (nodes not in valid clusters)
                    isolated_vectors = []
                    for c_id, v_idx in isolated_nodes:
                        isolated_vectors.append(client_codebooks[c_id][v_idx])
                    
                    if len(isolated_vectors) > 0:
                        # Randomly select from isolated vectors to fill the rest
                        num_to_fill = min(remaining_slots, len(isolated_vectors))
                        indices = torch.randperm(len(isolated_vectors))[:num_to_fill]
                        for i, idx in enumerate(indices):
                            new_codebook[num_aggregated + i] = isolated_vectors[idx]
                        self.logger.info(f"Filled {num_to_fill} slots with isolated vectors (from {len(isolated_vectors)} available).")
                
                elif self.codebook_fill_strategy == 'client_personalized':
                    # Client-personalized strategy: each client gets its own personalized vectors
                    # Returns a list of codebooks (one per client) instead of a single global codebook
                    self.logger.info(f"Using client_personalized strategy: each client will have personalized vectors in remaining {remaining_slots} slots.")
                    
                    # Build personalized codebooks for each client
                    personalized_codebooks = []
                    
                    for c_id in range(num_clients):
                        # Start with the shared aggregated vectors
                        client_codebook = new_codebook.clone()
                        
                        # Get this client's isolated nodes
                        client_isolated_nodes = [(cid, vid) for cid, vid in isolated_nodes if cid == c_id]
                        
                        if len(client_isolated_nodes) > 0:
                            # Calculate cross-client similarity and usage frequency for each isolated vector
                            # We want vectors with LOWEST similarity to other clients (most personalized)
                            # AND high usage frequency (not noise)
                            vector_similarities = []
                            
                            for node in client_isolated_nodes:
                                c_id_node, v_idx_node = node
                                vector = client_codebooks[c_id_node][v_idx_node]
                                
                                # Calculate average similarity to all OTHER clients' codebooks
                                total_sim = 0.0
                                count = 0
                                for other_c_id in range(num_clients):
                                    if other_c_id != c_id:
                                        # Normalize vectors for cosine similarity
                                        normalized_vector = F.normalize(vector.unsqueeze(0), p=2, dim=1)
                                        normalized_other_codebook = F.normalize(client_codebooks[other_c_id], p=2, dim=1)
                                        
                                        # Compute similarity with all vectors in other client's codebook
                                        similarities = torch.mm(normalized_vector, normalized_other_codebook.t())
                                        mean_sim = similarities.mean().item()
                                        total_sim += mean_sim
                                        count += 1
                                
                                avg_cross_client_sim = total_sim / count if count > 0 else 0.0
                                
                                # Get usage frequency for this vector
                                if client_usage_counts is not None and c_id < len(client_usage_counts):
                                    usage_count = client_usage_counts[c_id][v_idx_node].item()
                                else:
                                    usage_count = 1.0  # Default if usage not tracked
                                
                                # Store (node, vector, avg_similarity, usage_count)
                                vector_similarities.append((node, vector, avg_cross_client_sim, usage_count))
                            
                            # Normalize usage counts within this client's isolated vectors
                            max_usage = max(item[3] for item in vector_similarities)
                            
                            # Compute personalization score for each vector
                            scored_vectors = []
                            for node, vector, avg_sim, usage in vector_similarities:
                                # Normalize usage to [0, 1]
                                normalized_usage = usage / max_usage if max_usage > 0 else 0.0
                                
                                # Personalization score = -similarity + lambda * normalized_usage
                                # Lower cross-client similarity + higher usage = higher score
                                lambda_weight = 1.0  # Balance between similarity and usage (adjustable)
                                personalization_score = -avg_sim + lambda_weight * normalized_usage
                                
                                scored_vectors.append((node, vector, personalization_score, avg_sim, usage))
                            
                            # Sort by personalization score (descending): highest score = most personalized + most used
                            scored_vectors.sort(key=lambda x: x[2], reverse=True)
                            
                            # Select top remaining_slots most personalized vectors
                            num_to_select = min(remaining_slots, len(scored_vectors))
                            selected_vectors = [item[1] for item in scored_vectors[:num_to_select]]
                            
                            # Fill the remaining slots with selected personalized vectors
                            for i, vec in enumerate(selected_vectors):
                                client_codebook[num_aggregated + i] = vec
                            
                            # Log selection details
                            if num_to_select > 0:
                                top_score = scored_vectors[0]
                                bottom_score = scored_vectors[num_to_select - 1]
                                self.logger.info(
                                    f"Client {c_id}: Filled {num_to_select} personalized slots | "
                                    f"Top: score={top_score[2]:.3f} (sim={top_score[3]:.3f}, usage={top_score[4]:.0f}) | "
                                    f"Bottom: score={bottom_score[2]:.3f} (sim={bottom_score[3]:.3f}, usage={bottom_score[4]:.0f})"
                                )
                        else:
                            self.logger.warning(f"Client {c_id}: No isolated vectors available for personalization.")
                        
                        personalized_codebooks.append(client_codebook)
                    
                    # Return the list of personalized codebooks instead of a single global one
                    return personalized_codebooks
                
                elif self.codebook_fill_strategy == 'none':
                    # Do nothing, remaining slots stay as zeros (from torch.zeros_like initialization)
                    pass
                else:
                    self.logger.warning(f"Unknown codebook fill strategy: {self.codebook_fill_strategy}. Using 'none'.")
        
        # Visualize clustering results if save_dir is provided
        if save_dir is not None and round_num is not None:
            self._visualize_clustering_results(
                client_codebooks=client_codebooks,
                clusters=valid_clusters,  # Only pass valid clusters (size >= 2)
                visited=visited,
                adj=adj,  # Pass adjacency matrix for graph visualization
                round_num=round_num,
                save_dir=save_dir,
                client_names=client_names
            )
        
        return new_codebook
    
    def _visualize_clustering_results(self, client_codebooks, clusters, visited, adj, round_num, save_dir, client_names=None):
        """
        Visualize clustering results using a Grid Layout to clearly show cluster composition.
        Focuses on Multi-Client clusters where aggregation happens.
        """
        try:
            import networkx as nx
            import math
        except ImportError as e:
            self.logger.error(f"Missing required library for graph visualization: {e}")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        num_clients = len(client_codebooks)
        
        if client_names is None:
            client_names = [f"C{i}" for i in range(num_clients)]
        
        # 1. Filter and Sort Clusters
        # We are most interested in clusters that merge vectors from different clients.
        multi_client_clusters = []
        single_client_clusters = []
        
        for cluster in clusters:
            clients_in_cluster = set(c for c, _ in cluster)
            if len(clients_in_cluster) > 1:
                multi_client_clusters.append(cluster)
            else:
                single_client_clusters.append(cluster)
        
        # Sort multi-client clusters by size (descending)
        multi_client_clusters.sort(key=len, reverse=True)
        
        self.logger.info(f"Visualization processing {len(clusters)} total clusters:")
        self.logger.info(f"  - Multi-Client (Cross-client patterns): {len(multi_client_clusters)}")
        self.logger.info(f"  - Single-Client (Local/Isolated patterns): {len(single_client_clusters)}")

        # 2. Setup Grid Layout
        clusters_to_plot = multi_client_clusters
        title_prefix = f"Multi-Client Clusters (Found {len(multi_client_clusters)}/{len(clusters)})"
        
        if len(clusters_to_plot) == 0:
            self.logger.info("No multi-client clusters found to visualize. Plotting sample single-client clusters.")
            clusters_to_plot = single_client_clusters[:20] 
            title_prefix = f"Top 20 Single-Client Clusters (No Multi-Client Found in {len(clusters)})"

        num_plots = len(clusters_to_plot)
        if num_plots == 0:
             return

        cols = 5
        rows = math.ceil(num_plots / cols)
        
        # Limit figure size
        if rows > 50:
            self.logger.warning(f"Too many clusters ({num_plots}), truncating visualization to top 250.")
            clusters_to_plot = clusters_to_plot[:250]
            num_plots = len(clusters_to_plot)
            rows = math.ceil(num_plots / cols)

        fig_height = max(4, rows * 3)
        fig, axes = plt.subplots(rows, cols, figsize=(20, fig_height))
        
        if num_plots == 1:
            axes = [np.array([axes])] # Make it iterable
        else:
            axes = axes.flatten()
        
        # Color map for clients
        client_colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

        # 3. Draw each cluster
        for idx, cluster in enumerate(clusters_to_plot):
            ax = axes[idx] if isinstance(axes, np.ndarray) else axes[idx]
            
            # Create a small graph for this cluster
            G = nx.Graph()
            cluster_nodes_set = set(cluster)
            
            for client_id, vector_idx in cluster:
                node_id = f"{client_id}-{vector_idx}"
                # Label format: C{id}-{vec}
                label_str = f"C{client_id}-{vector_idx}"
                G.add_node(node_id, client_id=client_id, label=label_str)
            
            # Add edges from adj
            for node in cluster:
                if node in adj:
                    for neighbor in adj[node]:
                        if neighbor in cluster_nodes_set and node < neighbor:
                             n1 = f"{node[0]}-{node[1]}"
                             n2 = f"{neighbor[0]}-{neighbor[1]}"
                             G.add_edge(n1, n2)
            
            # Layout
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42, k=1.5) 
                
                # Node colors
                node_color_list = [client_colors[G.nodes[n]['client_id']] for n in G.nodes()]
                
                # Draw
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=400, alpha=0.9)
                nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
                
                # Draw labels
                labels = nx.get_node_attributes(G, 'label')
                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold')
            
            ax.set_title(f"Cluster {idx+1}\nSize: {len(cluster)}", fontsize=10)
            ax.axis('off')

        # Hide unused subplots
        if isinstance(axes, np.ndarray):
            for i in range(num_plots, len(axes)):
                axes[i].axis('off')

        # Global Title and Legend
        plt.suptitle(f"{title_prefix} - Round {round_num}\nThreshold: {self.similarity_threshold}", fontsize=16, y=1.00)
        
        # Create legend for clients
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=client_colors[i], markersize=10,
                                 label=client_names[i] if i < len(client_names) else f"Client {i}")
                          for i in range(num_clients)]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), title="Clients")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'grid_clustering_round{round_num}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved grid-based clustering visualization to {save_path}")
    
    
    
    # #############################################################################
    # Alternative optimized version using matrix operations
    # #############################################################################
    # def _aggregate_codebook_by_cos_similarity(self, client_codebooks):
    #     """
    #     Aggregates codebooks based on cosine similarity of individual code vectors.
    #     Optimized with matrix operations instead of nested loops.
    #     Only considers similarity across different clients, not within the same client.
    #     """
    #     self.logger.info(f"Server: Aggregating VQ codebook using cosine similarity strategy (Similarity threshold {self.similarity_threshold})...")
    #     num_clients = len(client_codebooks)
    #     num_codes, code_dim = client_codebooks[0].shape
        
    #     # 1. Concatenate all codebooks: (num_clients * num_codes, code_dim)
    #     all_codebooks = torch.cat(client_codebooks, dim=0)  # Shape: (num_clients * num_codes, code_dim)
        
    #     # 2. Normalize all vectors
    #     normalized_all = F.normalize(all_codebooks, p=2, dim=1)  # Shape: (num_clients * num_codes, code_dim)
        
    #     # 3. Compute full similarity matrix using matrix multiplication
    #     sim_matrix = torch.mm(normalized_all, normalized_all.t())  # Shape: (num_clients * num_codes, num_clients * num_codes)
        
    #     # 4. Zero out diagonal (self-similarity)
    #     mask_diagonal = torch.eye(sim_matrix.shape[0], device=self.device, dtype=torch.bool)
    #     sim_matrix = sim_matrix.masked_fill(mask_diagonal, 0)
        
    #     # 5. Create mask for same-client pairs and zero them out
    #     # For global index i, its client_id is i // num_codes
    #     # Two vectors are from the same client if (i // num_codes) == (j // num_codes)
    #     mask_same_client = torch.zeros_like(sim_matrix, dtype=torch.bool, device=self.device)
    #     for i in range(num_clients * num_codes):
    #         client_i = i // num_codes
    #         for j in range(num_clients * num_codes):
    #             client_j = j // num_codes
    #             if client_i == client_j:
    #                 mask_same_client[i, j] = True
        
    #     sim_matrix = sim_matrix.masked_fill(mask_same_client, 0)
        
    #     # 6. Find pairs with similarity > threshold
    #     similar_indices = torch.where(sim_matrix > self.similarity_threshold)
        
    #     # 7. Build adjacency list from indices
    #     adj = {i: [] for i in range(num_clients * num_codes)}
    #     for i, j in zip(*similar_indices):
    #         i_item, j_item = i.item(), j.item()
    #         if i_item < j_item:  # Avoid duplicate edges
    #             adj[i_item].append(j_item)
    #             adj[j_item].append(i_item)
        
    #     # 8. Find connected components using BFS
    #     visited = set()
    #     clusters = []
    #     for node in range(num_clients * num_codes):
    #         if node not in visited:
    #             cluster = []
    #             q = [node]
    #             visited.add(node)
    #             while q:
    #                 curr_node = q.pop(0)
    #                 cluster.append(curr_node)
    #                 for neighbor in adj[curr_node]:
    #                     if neighbor not in visited:
    #                         visited.add(neighbor)
    #                         q.append(neighbor)
    #             clusters.append(cluster)
        
    #     self.logger.info(f"Found {len(clusters)} clusters among {num_clients * num_codes} code vectors.")
        
    #     # 9. Aggregate vectors within each cluster using vectorized indexing
    #     aggregated_vectors = []
    #     for cluster in clusters:
    #         # Use advanced indexing to get all vectors in cluster at once
    #         vectors_in_cluster = all_codebooks[cluster]  # Shape: (cluster_size, code_dim)
    #         mean_vector = vectors_in_cluster.mean(dim=0)
    #         aggregated_vectors.append(mean_vector)
        
    #     # 10. Construct the new global codebook
    #     num_aggregated = len(aggregated_vectors)
        
    #     if num_aggregated >= num_codes:
    #         # Take the first `num_codes` aggregated vectors
    #         new_codebook = torch.stack(aggregated_vectors[:num_codes])
    #     else:
    #         # Fill with aggregated vectors first
    #         new_codebook = torch.zeros_like(client_codebooks[0])
    #         if num_aggregated > 0:
    #             new_codebook[:num_aggregated] = torch.stack(aggregated_vectors)
            
    #         # Fill remaining slots with isolated vectors
    #         remaining_slots = num_codes - num_aggregated
    #         if remaining_slots > 0:
    #             isolated_indices = [i for i in range(num_clients * num_codes) if i not in visited]
                
    #             if len(isolated_indices) > 0:
    #                 # Use vectorized indexing
    #                 isolated_vectors = all_codebooks[isolated_indices]
    #                 num_to_fill = min(remaining_slots, len(isolated_vectors))
    #                 perm_indices = torch.randperm(len(isolated_vectors), device=self.device)[:num_to_fill]
    #                 new_codebook[num_aggregated:num_aggregated + num_to_fill] = isolated_vectors[perm_indices]
        
    #     return new_codebook


    def _cos_similarity(self, client_updates, client_data_sizes, round_num=None, save_dir=None, client_names=None, client_usage_counts=None):
        """
        Aggregation via Cosine Similarity for VQ codebook.
        client_updates are expected to be a list of codebook tensors.
        
        Args:
            client_updates: List of codebook tensors
            client_data_sizes: List of client data sizes
            round_num: Current round number
            save_dir: Directory to save visualizations
            client_names: List of client names
            client_usage_counts: List of usage count tensors (for personalization)
        
        Returns:
            - For 'random_isolated' or 'none' strategies: OrderedDict with single global codebook
            - For 'client_personalized' strategy: List of OrderedDicts (one per client)
        """
        self.logger.info(f"Server: Aggregating using 'cos_similarity' strategy for codebooks (Similarity threshold {self.similarity_threshold})...")
        
        # --- Aggregate VQ Codebook using Cosine Similarity ---
        result = self._aggregate_codebook_by_cos_similarity(
            client_updates, 
            round_num=round_num, 
            save_dir=save_dir,
            client_names=client_names,
            client_data_sizes=client_data_sizes,
            client_usage_counts=client_usage_counts
        )
        
        # Check if result is a list (client_personalized) or a single tensor
        if isinstance(result, list):
            # client_personalized strategy: return list of state_dicts
            final_agg_weights = []
            for client_codebook in result:
                client_state_dict = OrderedDict()
                client_state_dict['_embedding.weight'] = client_codebook
                final_agg_weights.append(client_state_dict)
            return final_agg_weights
        else:
            # Other strategies: return single state_dict
            final_agg_weights = OrderedDict()
            final_agg_weights['_embedding.weight'] = result
            return final_agg_weights

    def aggregate(self, client_updates, client_data_sizes, round_num=None, save_dir=None, client_names=None, client_usage_counts=None):
        """
        Aggregates client model updates based on the chosen strategy.
        
        Args:
            client_updates: List of client model updates
            client_data_sizes: List of client data sizes
            round_num: Current round number (for visualization)
            save_dir: Directory to save visualizations
            client_names: List of client names
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
                round_num=round_num, 
                save_dir=save_dir,
                client_names=client_names,
                client_usage_counts=client_usage_counts
            )
            
            # Check if we have personalized codebooks
            if isinstance(agg_weights, list):
                # client_personalized: load the first client's codebook to global model (they share the same aggregated part)
                # but return the list so clients can get their personalized versions
                self.global_model.load_state_dict(agg_weights[0])
                self.logger.info(f"Server: Aggregation complete by '{self.strategy}' strategy with 'client_personalized' fill. "
                               f"Each client will receive personalized codebook.")
                return agg_weights  # Return list of personalized codebooks
            else:
                # Standard aggregation
                self.global_model.load_state_dict(agg_weights)
                self.logger.info(f"Server: Aggregation complete by '{self.strategy}' strategy. Global model updated.")
                return None  # No personalized weights
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

class Client:
    """
    A Client in a Federated Learning setup.
    It has its own local data and local models (decode/mustd).
    It receives the global model, trains on its data, and sends back the update.
    """
    def __init__(self, client_id, args, vqvae_config, device, logger):
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
        dim = vqvae_config['embedding_dim'] if not args.onehot else args.codebook_size
        
        # Get encoder and decoder configuration from args (loaded from Ablation_args.yaml)
        encoder_type = getattr(args, 'encoder_config_encoder_type', 'cnn')
        decoder_type = getattr(args, 'decoder_config_decoder_type', 'transformer')
        transformer_nhead = getattr(args, 'encoder_config_transformer_nhead', 4)
        transformer_layers = getattr(args, 'encoder_config_transformer_layers', 2)
        encoder_dropout = getattr(args, 'encoder_config_encoder_dropout', 0.1)
        rnn_layers = getattr(args, 'encoder_config_rnn_layers', 2)
        
        # Get VQVAE compression_factor
        self.compression_factor = vqvae_config['compression_factor']

        # Encoder is now a local model
        self.model_encoder = Encoder(
            in_channels=1,
            num_hiddens=vqvae_config['block_hidden_size'],
            num_residual_layers=vqvae_config['num_residual_layers'],
            num_residual_hiddens=vqvae_config['res_hidden_size'],
            embedding_dim=vqvae_config['embedding_dim'],
            compression_factor=vqvae_config['compression_factor'],
            encoder_type=encoder_type,
            seq_len=self.args.Tin,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            dropout=encoder_dropout,
            rnn_layers=rnn_layers
        ).to(self.device)

        # VQ is also local, but its codebook will be synced with the server
        self.model_vq = VectorQuantizer(
            num_embeddings=vqvae_config['num_embeddings'],
            embedding_dim=vqvae_config['embedding_dim'],
            commitment_cost=vqvae_config['commitment_cost']
        ).to(self.device)

        # Initialize codebook usage frequency counter for personalization
        self.codebook_usage_count = torch.zeros(
            vqvae_config['num_embeddings'], 
            device=self.device
        )

        self.model_decode = XcodeYtimeDecoder(
            d_in=dim,
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            seq_in_len=args.Tin // vqvae_config['compression_factor'],
            seq_out_len=args.Tout,
            dropout=0.0,
            decoder_type=decoder_type,
            compression_factor=vqvae_config['compression_factor'],
            num_residual_layers=vqvae_config['num_residual_layers'],
            num_residual_hiddens=vqvae_config['res_hidden_size'],
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

    def compute_codebook_diversity_loss(self, codebook):
        """
        Compute diversity loss for the codebook to encourage different vectors.
        
        Args:
            codebook: Tensor of shape (num_embeddings, embedding_dim)
        
        Returns:
            diversity_loss: Scalar tensor representing the diversity loss
        """
        if not getattr(self.args, 'codebook_diversity_enable_diversity_loss', False):
            return 0.0
        
        loss_type = getattr(self.args, 'codebook_diversity_diversity_loss_type', 'repulsion')
        temperature = getattr(self.args, 'codebook_diversity_diversity_temperature', 0.5)
        
        if loss_type == 'repulsion':
            # Repulsion loss: penalize vectors that are too similar
            # Normalize codebook vectors
            normalized_codebook = F.normalize(codebook, p=2, dim=1)  # (num_embeddings, embedding_dim)
            
            # Compute pairwise cosine similarity
            similarity_matrix = torch.mm(normalized_codebook, normalized_codebook.t())  # (num_embeddings, num_embeddings)
            
            # Zero out diagonal (self-similarity)
            mask = torch.eye(similarity_matrix.shape[0], device=self.device, dtype=torch.bool)
            similarity_matrix = similarity_matrix.masked_fill(mask, 0)
            
            # Repulsion loss: minimize positive similarities
            # Higher temperature makes the penalty stricter
            repulsion_loss = torch.sum(torch.clamp(similarity_matrix + temperature, min=0) ** 2)
            repulsion_loss = repulsion_loss / (similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1))
            
            return repulsion_loss
        
        elif loss_type == 'l2_distance':
            # L2 distance loss: maximize minimum pairwise distances
            # Compute pairwise L2 distances
            diff = codebook.unsqueeze(0) - codebook.unsqueeze(1)  # (num_embeddings, num_embeddings, embedding_dim)
            distances = torch.norm(diff, p=2, dim=2)  # (num_embeddings, num_embeddings)
            
            # Zero out diagonal
            mask = torch.eye(distances.shape[0], device=self.device, dtype=torch.bool)
            distances = distances.masked_fill(mask, float('inf'))
            
            # Loss: maximize minimum distance (minimize negative minimum distance)
            min_distances = torch.min(distances, dim=1)[0]  # (num_embeddings,)
            distance_loss = -torch.mean(min_distances)
            
            return distance_loss
        
        elif loss_type == 'cosine_similarity':
            # Cosine similarity based loss: minimize average similarity
            normalized_codebook = F.normalize(codebook, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_codebook, normalized_codebook.t())
            
            # Zero out diagonal
            mask = torch.eye(similarity_matrix.shape[0], device=self.device, dtype=torch.bool)
            similarity_matrix = similarity_matrix.masked_fill(mask, 0)
            
            # Loss: minimize mean absolute similarity
            similarity_loss = torch.mean(torch.abs(similarity_matrix))
            
            return similarity_loss
        
        else:
            self.logger.warning(f"Unknown diversity loss type: {loss_type}. Returning 0.")
            return 0.0

    def get_local_models(self):
        """Returns the local models of the client for checkpointing."""
        models = {
            f'client_{self.client_id}_{self.data_type}_encoder': self.model_encoder,
            f'client_{self.client_id}_{self.data_type}_decode': self.model_decode,
        }
        if self.model_mustd is not None:
            models[f'client_{self.client_id}_{self.data_type}_mustd'] = self.model_mustd
        return models

    def local_train(self, global_codebook_weights, vqvae_config):
        """
        Performs local training on the client's data.
        """
        if not self.dataloaders or 'train' not in self.dataloaders:
            self.logger.warning(f"Client {self.client_id}: No training data. Skipping local training.")
            return None

        self.logger.info(f"Client {self.client_id}: Starting local training for {self.args.local_epochs} epochs.")
        
        # Load the global codebook into the local VQ model
        self.model_vq.load_state_dict(global_codebook_weights)

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

        # Reset codebook usage counter before training
        self.codebook_usage_count.zero_()

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
                
                # Track codebook usage frequency (vectorized for efficiency)
                used_indices = encoding_indices.flatten()
                # Use scatter_add for fast batched updates
                self.codebook_usage_count.scatter_add_(0, used_indices, torch.ones_like(used_indices, dtype=self.codebook_usage_count.dtype))
                
                # 2. Use codes to predict with the local decode model
                xcodes = quantized_x.reshape(bs, x.shape[-1], vqvae_config['embedding_dim'], -1)
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

                # Add diversity loss
                if getattr(self.args, 'codebook_diversity_enable_diversity_loss', False):
                    diversity_loss = self.compute_codebook_diversity_loss(self.model_vq._embedding.weight)
                    diversity_weight = getattr(self.args, 'codebook_diversity_diversity_loss_weight', 0.1)
                    loss += diversity_loss * diversity_weight

                # Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(self.dataloaders['train'])
            self.logger.info(f"Client {self.client_id} | Epoch {epoch+1}/{self.args.local_epochs} | Avg Loss: {avg_epoch_loss:.4f}")

        # Log codebook usage statistics
        usage_stats = {
            'min': self.codebook_usage_count.min().item(),
            'max': self.codebook_usage_count.max().item(),
            'mean': self.codebook_usage_count.mean().item(),
            'std': self.codebook_usage_count.std().item(),
            'num_unused': (self.codebook_usage_count == 0).sum().item()
        }
        self.logger.info(f"Client {self.client_id} | Codebook usage stats: "
                        f"min={usage_stats['min']:.0f}, max={usage_stats['max']:.0f}, "
                        f"mean={usage_stats['mean']:.1f}, std={usage_stats['std']:.1f}, "
                        f"unused_codes={usage_stats['num_unused']}/{len(self.codebook_usage_count)}")

        # Return only the updated codebook tensor
        return self.model_vq._embedding.weight.detach()

    def evaluate(self, global_codebook_weights, vqvae_config, split='val', use_local_codebook=False):
        """
        Evaluates the current global model on the client's local data.
        
        Args:
            global_codebook_weights: Global codebook weights from server (can be None if use_local_codebook=True)
            vqvae_config: VQ-VAE configuration
            split: 'val' for validation, 'test' for final testing
            use_local_codebook: If True, use the client's own local codebook instead of global
        """
        if split not in self.dataloaders:
            self.logger.warning(f"Client {self.client_id}: No '{split}' dataloader available.")
            return None

        self.logger.info(f"Client {self.client_id}: Evaluating on '{split}' data{' (using local codebook)' if use_local_codebook else ''}.")

        # Load the global codebook into the local VQ model (unless using local codebook)
        if not use_local_codebook and global_codebook_weights is not None:
            self.model_vq.load_state_dict(global_codebook_weights)
        elif use_local_codebook:
            self.logger.info(f"Client {self.client_id}: Using local codebook for evaluation (no aggregation).")
        else:
            self.logger.warning(f"Client {self.client_id}: No global codebook provided and use_local_codebook=False.")

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
                xcodes = quantized_x.reshape(bs, x.shape[-1], vqvae_config['embedding_dim'], -1)
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
