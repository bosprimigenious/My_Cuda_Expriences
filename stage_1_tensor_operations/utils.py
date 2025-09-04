import torch

def describe_tensor(t: torch.Tensor, name="Tensor"):
    print(f"--- {name} ---")
    print("Shape:", t.shape)
    print("Device:", t.device)
    print("Dtype:", t.dtype)
    print("Values:\n", t)
