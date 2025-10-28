import torch
print(torch.cuda.is_available())   # Should be True if GPU is accessible
print(torch.version.cuda)          # CUDA version PyTorch was built with
print(torch.cuda.device_count())   # How many GPUs PyTorch sees
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")