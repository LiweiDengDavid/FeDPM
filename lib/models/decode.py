import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from lib.models.Encoder_and_Retrieval import ResidualStack

"""
This files contains helper models that convert codes to time. 

* Transformer: Is a transformer model that takes in as input a sequence of length N 
               of D-dimensional codes (N, B, D) and predicts a N-length sequence in time (N, B, C)
               where C is the compression factor (aka each element in the sequence predict C time steps)

* MuStdModel: Is a simple MLP that takes as input the past time series (B, Tin) 
                and predicts the mean and std of the future time series
"""


def conv_lout(lin, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor(
        (lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class ResMLPBlock(nn.Module):
    def __init__(self, hidden_dim=128, res_dim=128):
        super(ResMLPBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, res_dim)
        self.fc2 = nn.Linear(res_dim, hidden_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        x = F.relu(x + y)
        return x


class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, res_dim, nblocks):
        super(ResMLP, self).__init__()

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            ResMLPBlock(hidden_dim=hidden_dim, res_dim=res_dim) for _ in range(nblocks)
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.fc_out(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout=0.0):
        super(SimpleMLP, self).__init__()

        self.nlayers = len(hidden_dims)

        layers = []
        dim = in_dim
        for i in range(self.nlayers):
            layer = nn.Linear(dim, hidden_dims[i])
            layers.append(layer)
            dim = hidden_dims[i]
        self.fcs = nn.ModuleList(layers)
        self.fc_out = nn.Linear(dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc_out(x)

        return x


class SimpleConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(SimpleConv1d, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=2)
        lout = conv_lout(in_dim, kernel_size=4, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2)
        lout = conv_lout(lout, kernel_size=4, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)
        lout = conv_lout(lout, kernel_size=4, stride=2)

        dim = lout * 256
        fcs = []
        for hidden_dim in hidden_dims:
            fcs.append(nn.Linear(dim, hidden_dim))
            dim = hidden_dim
        self.linears = nn.ModuleList(fcs)
        self.linear_out = nn.Linear(dim, 2)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, Tin)
        Returns:
            out: tensor of shape (B, 2)
        """
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        for fc in self.linears:
            x = F.relu(fc(x))
        x = self.linear_out(x)
        return x


class MuStdModel(nn.Module):
    def __init__(self, Tin, Tout, hidden_dims, dropout=0.0, is_mlp=True):
        super(MuStdModel, self).__init__()

        # mean, std
        if is_mlp:
            self.mustd = SimpleMLP(
                in_dim=Tin + 2, out_dim=2, hidden_dims=hidden_dims, dropout=dropout
            )
        else:
            raise NotImplementedError

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: of shape (batch_size, Tin)
        Output:
            out: of shape (batch_size, 2)
        """
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = torch.cat((x, mean, std), dim=1)
        x = self.mustd(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        seq_len: int = 5000,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super(Transformer, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear_out = nn.Linear(d_model, d_out)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, codes: torch.Tensor, codes_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Arguments:
            time: tensor of shape (batch_size, Tin)
            codes: tensor of shape (seq_len, batch_size, d_in)
            src_mask: tensor of shape (seq_len, seq_len)

        Returns:
            time_toutput: tensor of shape (batch_size, 2)
            codes_output: tensor of shape (seq_len, batch_size, dout)
        """
        if self.has_linear_in:
            codes = self.linear_in(codes)
        codes = self.pos_encoder(codes)
        codes_output = self.transformer_encoder(codes, codes_mask)
        codes_output = self.linear_out(codes_output)  # (seq, batch, dout)

        return codes_output


# Decoder Backbone Classes
class TransformerDecoderBackbone(nn.Module):
    """Transformer-based decoder backbone."""
    def __init__(self, d_model, nhead, d_hid, nlayers, seq_in_len, dropout=0.0, norm_first=False):
        super(TransformerDecoderBackbone, self).__init__()
        self.d_model = d_model
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_in_len)
        
        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (seq_in_len, batch_size, d_model)
            x_mask: tensor of shape (seq_in_len, seq_in_len)
        
        Returns:
            output: tensor of shape (seq_in_len, batch_size, d_model)
        """
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, x_mask)
        return x


class CNNDecoderBackbone(nn.Module):
    """CNN-based decoder backbone with transpose convolutions."""
    def __init__(self, d_model, d_hid, compression_factor, num_residual_layers=2, 
                 num_residual_hiddens=32, seq_in_len=96):
        super(CNNDecoderBackbone, self).__init__()
        self.compression_factor = compression_factor
        self.d_model = d_model
        self.seq_in_len = seq_in_len
        
        # Initial conv to project from embedding space
        self._conv_1 = nn.Conv1d(in_channels=d_model,
                                 out_channels=d_hid,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=d_hid,
                                            num_hiddens=d_hid,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)
        
        if compression_factor == 4:
            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=d_hid // 2,
                                                    out_channels=d_model,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
        
        elif compression_factor == 8:
            self._conv_trans_A = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=d_hid // 2,
                                                    out_channels=d_model,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
        
        elif compression_factor == 12:
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid,
                                                    kernel_size=5,
                                                    stride=3, padding=1)
            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=d_hid // 2,
                                                    out_channels=d_model,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
        
        elif compression_factor == 16:
            self._conv_trans_A = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_B = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=d_hid,
                                                    out_channels=d_hid // 2,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=d_hid // 2,
                                                    out_channels=d_model,
                                                    kernel_size=4,
                                                    stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch_size, d_model, seq_in_len)
        
        Returns:
            output: tensor of shape (batch_size, d_model, seq_in_len)
        """
        x = self._conv_1(x)
        x = self._residual_stack(x)
        
        if self.compression_factor == 4:
            x = F.relu(self._conv_trans_1(x))
            x = F.relu(self._conv_trans_2(x))
        
        elif self.compression_factor == 8:
            x = F.relu(self._conv_trans_A(x))
            x = F.relu(self._conv_trans_1(x))
            x = F.relu(self._conv_trans_2(x))
        
        elif self.compression_factor == 12:
            x = F.relu(self._conv_trans_2(x))
            x = F.relu(self._conv_trans_3(x))
            x = F.relu(self._conv_trans_4(x))
        
        elif self.compression_factor == 16:
            x = F.relu(self._conv_trans_A(x))
            x = F.relu(self._conv_trans_B(x))
            x = F.relu(self._conv_trans_1(x))
            x = F.relu(self._conv_trans_2(x))
        
        return x


class RNNDecoderBackbone(nn.Module):
    """RNN-based decoder backbone with LSTM or GRU."""
    def __init__(self, d_model, d_hid, nlayers, seq_in_len, dropout=0.0, rnn_type='lstm'):
        super(RNNDecoderBackbone, self).__init__()
        self.d_model = d_model
        self.d_hid = d_hid
        self.rnn_type = rnn_type.lower()
        
        # Input projection
        self.input_projection = nn.Linear(d_model, d_hid)
        
        # RNN layers
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=nlayers,
                batch_first=False,
                dropout=dropout if nlayers > 1 else 0.0,
                bidirectional=False
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=d_hid,
                hidden_size=d_hid,
                num_layers=nlayers,
                batch_first=False,
                dropout=dropout if nlayers > 1 else 0.0,
                bidirectional=False
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")
        
        # Output projection back to d_model
        self.output_projection = nn.Linear(d_hid, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (seq_in_len, batch_size, d_model)
        
        Returns:
            output: tensor of shape (seq_in_len, batch_size, d_model)
        """
        # Project input to d_hid
        x = self.input_projection(x)  # (seq_in_len, batch_size, d_hid)
        
        # Apply RNN
        x, _ = self.rnn(x)  # (seq_in_len, batch_size, d_hid)
        
        # Project back to d_model
        x = self.output_projection(x)  # (seq_in_len, batch_size, d_model)
        
        return x


class FCDecoderBackbone(nn.Module):
    """Fully-connected layer decoder backbone."""
    def __init__(self, d_model, d_hid, nlayers, seq_in_len, dropout=0.0):
        super(FCDecoderBackbone, self).__init__()
        
        fc_layers = []
        input_dim = d_model * seq_in_len
        
        for i in range(nlayers):
            if i == 0:
                fc_layers.append(nn.Linear(input_dim, d_hid))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
            elif i == nlayers - 1:
                fc_layers.append(nn.Linear(d_hid, input_dim))
            else:
                fc_layers.append(nn.Linear(d_hid, d_hid))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.seq_in_len = seq_in_len
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (seq_in_len, batch_size, d_model)
        
        Returns:
            output: tensor of shape (seq_in_len, batch_size, d_model)
        """
        # Convert from (seq_in_len, batch_size, d_model) to (batch_size, seq_in_len, d_model)
        x = torch.permute(x, (1, 0, 2))
        # Flatten to (batch_size, seq_in_len * d_model)
        batch_size = x.shape[0]
        x = x.flatten(start_dim=1)
        # Apply FC layers
        x = self.fc_layers(x)
        # Reshape back to (batch_size, seq_in_len, d_model)
        x = x.reshape(batch_size, self.seq_in_len, self.d_model)
        # Convert to (seq_in_len, batch_size, d_model)
        x = torch.permute(x, (1, 0, 2))
        
        return x


class XcodeYtimeDecoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        seq_in_len: int = 96,
        seq_out_len: int = 96,
        dropout: float = 0.0,
        batch_first: bool = False,
        norm_first: bool = False,
        decoder_type: str = 'fc',
        compression_factor: int = 4,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 64,
    ):
        super(XcodeYtimeDecoder, self).__init__()
       
        self.d_model = d_model
        self.decoder_type = decoder_type.lower()
        self.seq_in_len = seq_in_len
        self.seq_out_len = seq_out_len

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        # Select decoder backbone based on decoder_type
        if self.decoder_type == 'transformer':
            self.backbone = TransformerDecoderBackbone(
                d_model=d_model,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                seq_in_len=seq_in_len,
                dropout=dropout,
                norm_first=norm_first
            )
        
        elif self.decoder_type == 'cnn':
            self.backbone = CNNDecoderBackbone(
                d_model=d_model,
                d_hid=d_hid,
                compression_factor=compression_factor,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                seq_in_len=seq_in_len
            )
        
        elif self.decoder_type == 'lstm':
            self.backbone = RNNDecoderBackbone(
                d_model=d_model,
                d_hid=d_hid,
                nlayers=nlayers,
                seq_in_len=seq_in_len,
                dropout=dropout,
                rnn_type='lstm'
            )
        
        elif self.decoder_type == 'gru':
            self.backbone = RNNDecoderBackbone(
                d_model=d_model,
                d_hid=d_hid,
                nlayers=nlayers,
                seq_in_len=seq_in_len,
                dropout=dropout,
                rnn_type='gru'
            )
        
        else:
            # Default to FC backbone
            self.backbone = FCDecoderBackbone(
                d_model=d_model,
                d_hid=d_hid,
                nlayers=nlayers,
                seq_in_len=seq_in_len,
                dropout=dropout
            )
        if self.decoder_type != 'cnn':
            # For non-CNN backbones, we need to project from (seq_in_len * d_model) to seq_out_len
            self.linear_out = nn.Linear(d_model * seq_in_len, seq_out_len)
        else:
            # For CNN backbone, project from (d_model * seq_in_len * compression_factor) to seq_out_len
            self.linear_out = nn.Linear(d_model * seq_in_len * compression_factor, seq_out_len)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the decoder model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape (seq_in_len, batch_size, d_in)
            x_mask: tensor of shape (seq_in_len, seq_in_len)

        Returns:
            y: tensor of shape (batch_size, seq_out_len)
        """
        if self.has_linear_in:
            x = self.linear_in(x)
        
        # Process through backbone
        # x shape (seq_in_len, batch_size, d_model)
        if self.decoder_type == 'cnn':
            # CNN backbone expects (batch_size, d_model, seq_in_len)
            x = torch.permute(x, (1, 2, 0))  # (batch_size, d_model, seq_in_len)
            x = self.backbone(x)  # (batch_size, d_model, seq_in_len)
            x = torch.permute(x, (2, 0, 1))  # (seq_in_len, batch_size, d_model)
        elif self.decoder_type in ['transformer']:
            x = self.backbone(x, x_mask)  # (seq_in_len, batch_size, d_model)
        else:
            # RNN, FC backbones expect (seq_in_len, batch_size, d_model)
            x = self.backbone(x)  # (seq_in_len, batch_size, d_model)
        
        # Convert to (batch_size, seq_in_len * d_model) for final projection
        x = torch.permute(x, (1, 0, 2))  # (batch_size, seq_in_len, d_model)
        x = x.flatten(start_dim=1)  # (batch_size, seq_in_len * d_model)
        
        # Project to output sequence length
        y = self.linear_out(x)  # (batch_size, seq_out_len)

        return y