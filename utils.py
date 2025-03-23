import torch

def random_quaternion(batch_size=1):
    """Sample uniform random quaternions using Marsaglia method."""
    u1 = torch.rand(batch_size)
    u2 = torch.rand(batch_size)
    u3 = torch.rand(batch_size)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return torch.stack([q1, q2, q3, q4], dim = -1)