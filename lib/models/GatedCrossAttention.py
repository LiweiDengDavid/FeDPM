import torch
from torch import nn

class GatedCrossAttention(nn.Module):
    def __init__(self, attn, hidden_size):
        super().__init__()
        self.attn = attn  # Original Attention layer (use the original pre-trained weight) (e.g., GPT2Attention)
        self.attn_gate = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.]), requires_grad=True)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None):
        # x: (B, T, D) text embedding
        # encoder_hidden_states: (B, S, D) time series embedding
        # attention_mask: (B, T)
        # encoder_attention_mask: (B, S)
        # cross attention: Q=x, K/V=encoder_hidden_states
        attn_out = self.attn(
            hidden_states=x,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )[0]
        x = attn_out * self.attn_gate.tanh() + x
        ff_out = self.ff(x)
        x = ff_out * self.ff_gate.tanh() + x
        return x