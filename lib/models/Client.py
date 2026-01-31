
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import os
import json
from lib.utils.results_logger import human_count_params
from FeDPM.lib.models.decoder import XcodeYtimeDecoder, MuStdModel
from lib.models.revin import RevIN
from lib.models.metrics import pearsoncor
from lib.utils.data_utils import create_time_series_dataloader, get_params, loss_fn
from lib.models.Encoder_and_Retrieval import PMR, Encoder # Import Encoder and PMR (Prototypical Memories Retrieval)

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
        encoder_type = getattr(args, 'encoder_config_encoder_type', 'transformer')
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
        self.model_pmr = PMR(
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
        
        # Load the global memory into the local PMR model
        self.model_pmr.load_state_dict(global_memory_weights)

        # Set models to train mode
        self.model_encoder.train()
        self.model_pmr.train()
        self.model_decode.train()
        if self.model_mustd is not None:
            self.model_mustd.train()

        # Optimizer for all local parameters
        all_params = (
            list(self.model_encoder.parameters()) +
            list(self.model_pmr.parameters()) +
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
                vq_loss, quantized_x, perplexity, encodings, encoding_indices, _ = self.model_pmr(latent_x)
                
                # Track memory usage frequency for personalization
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

                    # Total loss includes PMR loss
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

        # Return only the updated memory tensor
        return self.model_pmr._embedding.weight.detach()

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
            self.model_pmr.load_state_dict(global_memory_weights)
        elif use_local_memory:
            self.logger.info(f"Client {self.client_id}: Using local memory for evaluation (no aggregation).")
        else:
            self.logger.warning(f"Client {self.client_id}: No global memory provided and use_local_memory=False.")

        # Set all models to evaluation mode
        self.model_encoder.eval()
        self.model_pmr.eval()
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
                
                # --- Forward pass ---
                if self.model_mustd is not None:
                    # Normalize input x using revin_in
                    norm_x = self.model_mustd.revin_in(x, "norm")
                    # For loss calculation only, normalize y 
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
                vq_loss, quantized_x, _, _, _, _ = self.model_pmr(latent_x)
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
