import torch

def add_noise_to_memory(memory, noise_type, noise_scale, device):
    """
    Add differential privacy noise to memory tensor.
    
    Args:
        memory: Tensor of shape (num_embeddings, embedding_dim)
        noise_type: Type of noise ('laplace', 'gaussian', 'exponential', or 'none')
        noise_scale: Scale factor for noise (privacy budget)
        device: Device to create noise tensors on
        
    Returns:
        Noisy memory tensor with same shape as input
    """
    if noise_type == 'none':
        return memory
    
    num_embeddings, embedding_dim = memory.shape
    
    if noise_type == 'laplace':
        # Laplace distribution: μ=0, λ=1
        # In PyTorch, Laplace is parameterized by (loc, scale) where scale = 1/λ
        # So for λ=1, we use scale=1/noise_scale
        laplace_dist = torch.distributions.Laplace(0, 1.0 / noise_scale)
        noise = laplace_dist.sample(memory.shape).to(device)
        
    elif noise_type == 'gaussian':
        # Gaussian distribution: μ=0, σ=1
        # In PyTorch, Normal is parameterized by (mean, std)
        # So for σ=1, we use std=noise_scale
        gaussian_dist = torch.distributions.Normal(0, noise_scale)
        noise = gaussian_dist.sample(memory.shape).to(device)
        
    elif noise_type == 'exponential':
        # Exponential distribution: λ=1
        # In PyTorch, Exponential is parameterized by rate (λ)
        # So for λ=1, we use rate=1/noise_scale
        exponential_dist = torch.distributions.Exponential(1.0 / noise_scale)
        noise = exponential_dist.sample(memory.shape).to(device)
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    noisy_memory = memory + noise
    return noisy_memory
