import torch
import torch.nn as nn
import torch.nn.functional as F


# #############################################################################
# Encoder Base Modules and Architecture Implementations
# #############################################################################

class CNNEncoderBackbone(nn.Module):
    """CNN-based encoder backbone for VQ-VAE."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 embedding_dim, compression_factor):
        super(CNNEncoderBackbone, self).__init__()
        self.compression_factor = compression_factor
        self.embedding_dim = embedding_dim
        
        if compression_factor == 4:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)
            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 8:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)
            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 12:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=3, padding=1)
            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)
            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        elif compression_factor == 16:
            self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                     out_channels=num_hiddens // 2,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=4,
                                     stride=2, padding=1)
            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
                                     out_channels=num_hiddens,
                                     kernel_size=3,
                                     stride=1, padding=1)
            self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                                 num_hiddens=num_hiddens,
                                                 num_residual_layers=num_residual_layers,
                                                 num_residual_hiddens=num_residual_hiddens)
            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        """Forward pass for CNN encoder backbone."""
        compression_factor = self.compression_factor
        
        if compression_factor == 4:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = self._conv_1(x)
            x = F.relu(x)
            x = self._conv_2(x)
            x = F.relu(x)
            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 8:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = self._conv_1(x)
            x = F.relu(x)
            x = self._conv_2(x)
            x = F.relu(x)
            x = self._conv_A(x)
            x = F.relu(x)
            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 12:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = self._conv_1(x)
            x = F.relu(x)
            x = self._conv_2(x)
            x = F.relu(x)
            x = self._conv_3(x)
            x = F.relu(x)
            x = self._conv_4(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x

        elif compression_factor == 16:
            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
            x = self._conv_1(x)
            x = F.relu(x)
            x = self._conv_2(x)
            x = F.relu(x)
            x = self._conv_A(x)
            x = F.relu(x)
            x = self._conv_B(x)
            x = F.relu(x)
            x = self._conv_3(x)
            x = self._residual_stack(x)
            x = self._pre_vq_conv(x)
            return x


class FCEncoderBackbone(nn.Module):
    """Fully Connected encoder backbone with patching (PatchTST-like)."""
    def __init__(self, seq_len, patch_len, embedding_dim, num_hiddens, dropout=0.1):
        super(FCEncoderBackbone, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = patch_len  # Non-overlapping patches
        self.embedding_dim = embedding_dim
        
        # Calculate patch_num from seq_len and patch_len
        self.patch_num = seq_len // patch_len
        
        # Patch embedding: FC layer to map patch_len -> embedding_dim
        self.patch_embedding = nn.Linear(patch_len, embedding_dim)
        
        # FC layers for further processing
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, num_hiddens),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_hiddens, embedding_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, inputs):
        """
        Forward pass for FC encoder.
        Args:
            inputs: (batch_size, seq_len) time series
        Returns:
            output: (batch_size, patch_num, embedding_dim)
        """
        batch_size,seq_len = inputs.shape
        
        if seq_len <= self.patch_len:
            ts_pad_num = self.patch_len - seq_len
        else:
            if seq_len % self.stride == 0:
                ts_pad_num = 0
            else:
                ts_pad_num = (seq_len // self.stride) * self.stride + self.patch_len - seq_len

        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        x_inp = ts_padding(inputs)

        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Embed patches through FC layer
        patch_embeddings = self.patch_embedding(x_inp)  # (batch_size, patch_num, embedding_dim)
        
        # Process through FC layers
        output = self.fc_layers(patch_embeddings)  # (batch_size, patch_num, embedding_dim)
        
        # Reshape to (batch_size, embedding_dim, patch_num) for compatibility with VQ
        output = output.permute(0, 2, 1)  # (batch_size, embedding_dim, patch_num)
        
        return output


class TransformerEncoderBackbone(nn.Module):
    """Transformer-based encoder backbone with patching (PatchTST-like)."""
    def __init__(self, seq_len, patch_len, embedding_dim, num_hiddens, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerEncoderBackbone, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = patch_len  # Non-overlapping patches
        self.embedding_dim = embedding_dim
        
        # Calculate patch_num from seq_len and patch_len
        self.patch_num = seq_len // patch_len
        
        # Patch embedding: FC layer to map patch_len -> embedding_dim
        self.patch_embedding = nn.Linear(patch_len, embedding_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.patch_num, embedding_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=num_hiddens,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        """
        Forward pass for Transformer encoder.
        Args:
            inputs: (batch_size, seq_len) time series
        Returns:
            output: (batch_size, embedding_dim, patch_num)
        """
        batch_size,seq_len = inputs.shape
        
        if seq_len <= self.patch_len:
            ts_pad_num = self.patch_len - seq_len
        else:
            if seq_len % self.stride == 0:
                ts_pad_num = 0
            else:
                ts_pad_num = (seq_len // self.stride) * self.stride + self.patch_len - seq_len

        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        x_inp = ts_padding(inputs)

        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embed patches through FC layer
        patch_embeddings = self.patch_embedding(x_inp)  # (batch_size, patch_num, embedding_dim)
        
        # Add positional encoding
        patch_embeddings = patch_embeddings + self.positional_encoding
        patch_embeddings = self.dropout(patch_embeddings)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(patch_embeddings)  # (batch_size, patch_num, embedding_dim)
        
        # Reshape to (batch_size, embedding_dim, patch_num) for compatibility with VQ
        output = transformer_output.permute(0, 2, 1)  # (batch_size, embedding_dim, patch_num)
        
        return output


class RNNEncoderBackbone(nn.Module):
    """RNN-based encoder backbone with patching."""
    def __init__(self, seq_len, patch_len, embedding_dim, num_hiddens, num_layers=2, dropout=0.1, rnn_type='lstm'):
        super(RNNEncoderBackbone, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = patch_len  # Non-overlapping patches
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type.lower()
        
        # Calculate patch_num from seq_len and patch_len
        self.patch_num = seq_len // patch_len
        
        # Patch embedding: FC layer to map patch_len -> embedding_dim
        self.patch_embedding = nn.Linear(patch_len, embedding_dim)
        
        # RNN encoder (LSTM or GRU)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}. Supported: 'lstm', 'gru'")
        
        # Project RNN output to embedding_dim
        rnn_output_dim = num_hiddens * 2  # bidirectional
        self.output_projection = nn.Linear(rnn_output_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        """
        Forward pass for RNN encoder.
        Args:
            inputs: (batch_size, seq_len) time series
        Returns:
            output: (batch_size, embedding_dim, patch_num)
        """
        batch_size, seq_len = inputs.shape
        
        if seq_len <= self.patch_len:
            ts_pad_num = self.patch_len - seq_len
        else:
            if seq_len % self.stride == 0:
                ts_pad_num = 0
            else:
                ts_pad_num = (seq_len // self.stride) * self.stride + self.patch_len - seq_len

        ts_padding = nn.ReplicationPad1d((0, ts_pad_num))
        x_inp = ts_padding(inputs)

        x_inp = x_inp.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embed patches through FC layer
        patch_embeddings = self.patch_embedding(x_inp)  # (batch_size, patch_num, embedding_dim)
        
        # Apply RNN
        rnn_output, _ = self.rnn(patch_embeddings)  # (batch_size, patch_num, num_hiddens*2)
        
        # Project to embedding_dim
        output = self.output_projection(rnn_output)  # (batch_size, patch_num, embedding_dim)
        output = self.dropout(output)
        
        # Reshape to (batch_size, embedding_dim, patch_num) for compatibility with VQ
        output = output.permute(0, 2, 1)  # (batch_size, embedding_dim, patch_num)
        
        return output


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    """Unified Encoder that supports multiple architectures: CNN, FC, Transformer, LSTM, and GRU."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 embedding_dim, compression_factor, encoder_type='cnn', seq_len=96, 
                 transformer_nhead=4, transformer_layers=2, dropout=0.1, rnn_layers=2):
        """
        Args:
            in_channels: Input channels (usually 1 for time series)
            num_hiddens: Number of hidden units
            num_residual_layers: Number of residual layers (for CNN)
            num_residual_hiddens: Number of residual hidden units (for CNN)
            embedding_dim: Embedding dimension for VQ
            compression_factor: Compression factor (patch_len for FC/Transformer/LSTM/GRU, stride for CNN)
            encoder_type: Type of encoder ('cnn', 'fc', 'transformer', 'lstm', 'gru')
            seq_len: Sequence length (used for FC/Transformer/LSTM/GRU)
            transformer_nhead: Number of heads for Transformer
            transformer_layers: Number of Transformer layers
            dropout: Dropout rate for FC/Transformer/LSTM/GRU
            rnn_layers: Number of RNN layers (for LSTM/GRU)
        """
        super(Encoder, self).__init__()
        
        self.encoder_type = encoder_type.lower()
        self.compression_factor = compression_factor
        self.embedding_dim = embedding_dim
        
        if self.encoder_type == 'cnn':
            self.backbone = CNNEncoderBackbone(
                in_channels=in_channels,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                embedding_dim=embedding_dim,
                compression_factor=compression_factor
            )
        
        elif self.encoder_type == 'fc':
            self.backbone = FCEncoderBackbone(
                seq_len=seq_len,
                patch_len=compression_factor,
                embedding_dim=embedding_dim,
                num_hiddens=num_hiddens,
                dropout=dropout
            )
        
        elif self.encoder_type == 'transformer':
            self.backbone = TransformerEncoderBackbone(
                seq_len=seq_len,
                patch_len=compression_factor,
                embedding_dim=embedding_dim,
                num_hiddens=num_hiddens,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
                dropout=dropout
            )
        
        elif self.encoder_type in ['lstm', 'gru']:
            self.backbone = RNNEncoderBackbone(
                seq_len=seq_len,
                patch_len=compression_factor,
                embedding_dim=embedding_dim,
                num_hiddens=num_hiddens,
                num_layers=rnn_layers,
                dropout=dropout,
                rnn_type=self.encoder_type
            )
        
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}. "
                           f"Supported types: 'cnn', 'fc', 'transformer', 'lstm', 'gru'")

    def forward(self, inputs, compression_factor=None):
        """
        Forward pass for the encoder.
        Args:
            inputs: Input tensor (batch_size, seq_len) or (batch_size, 1, seq_len) for CNN
            compression_factor: Optional override for compression factor (mainly for backward compatibility)
        Returns:
            output: Encoded representation (batch_size, embedding_dim, patch_num/compressed_len)
        """
        return self.backbone(inputs)




class PMR(nn.Module):
    """
    PMR: Prototypical Memories Retrieval
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(PMR, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances (speed up for L2)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)  # one-hot encoding

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach() # TODO Straight Through Estimator (Trick of training VQ-VAE)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # quantities the whole batch perplexity
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings



