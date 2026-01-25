import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os

def visualize_and_save_gate_values(clients, stage_name, save_path, logger):
    """
    Visualizes and saves the gate parameters (attn_gate, ff_gate) and their tanh values for each client.
    A separate plot is created for each client.
    """
    logger.info("Visualizing and saving gate parameters for all clients...")
    for client in clients:
        try:
            # Handle DataParallel wrapper
            llm_predictor = client.llm_predictor.module if isinstance(client.llm_predictor, nn.DataParallel) else client.llm_predictor
            
            num_layers = len(llm_predictor.gated_blocks)
            layer_indices = range(num_layers)
            
            attn_gate_vals = []
            attn_gate_tanh_vals = []

            for block in llm_predictor.gated_blocks:
                # Extract scalar values from the parameters
                attn_gate_vals.append(block.attn_gate.item())
                attn_gate_tanh_vals.append(torch.tanh(block.attn_gate).item())


            # Create a plot with two subplots
            fig, axs = plt.subplots(2, 1, figsize=(14, 12))
            fig.suptitle(f'Client {client.client_id} ({client.data_type}) - Gate Values at Stage: {stage_name}', fontsize=16)

            # Subplot 1: Raw gate values
            axs[0].plot(layer_indices, attn_gate_vals, marker='o', linestyle='-', label='attn_gate (raw)')
            axs[0].set_title('Raw Gate Parameter Values')
            axs[0].set_xlabel('Layer Index')
            axs[0].set_ylabel('Value')
            axs[0].legend()
            axs[0].grid(True)

            # Subplot 2: Tanh activated gate values
            axs[1].plot(layer_indices, attn_gate_tanh_vals, marker='o', linestyle='-', label='tanh(attn_gate)')
            axs[1].set_title('Tanh Activated Gate Values')
            axs[1].set_xlabel('Layer Index')
            axs[1].set_ylabel('Value')
            axs[1].set_ylim(-1.05, 1.05)  # Tanh values are in [-1, 1]
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Save the figure
            plot_filename = f'client_{client.client_id}_gate_values_{stage_name}.png'
            plot_save_path = os.path.join(save_path, plot_filename)
            plt.savefig(plot_save_path)
            plt.close(fig) # Close the figure to free up memory
            logger.info(f"Saved gate visualization for Client {client.client_id} to {plot_save_path}")

        except Exception as e:
            logger.error(f"Failed to visualize gates for Client {client.client_id}: {e}", exc_info=True)
