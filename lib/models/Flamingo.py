import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops_exts import rearrange_many

# Thanks to Code: https://github.com/lucidrains/flamingo-pytorch

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
        # self.ff_gate = nn.Parameter(torch.tensor([0.]), requires_grad=True)

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
        # x = ff_out * self.ff_gate.tanh() + x
        return ff_out
    

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )



class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, projection_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)