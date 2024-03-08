import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Using torch.Tensor with dtype parameter directly
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                         dtype=torch.float32,
                         device=device,
                         requires_grad=True
                         )

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization method
x1 = torch.empty(size=(3, 3))
x2 = torch.zeros(size=(3, 3))
x3 = torch.rand(size=(3, 3))
x4 = torch.ones(size=(3, 3))
x5 = torch.eye(5, 5)
x6 = torch.arange(start=0, end=5, step=1)
x7 = torch.linspace(start=.1, end=1, steps=10)
x8 = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x9 = torch.empty(size=(1, 5)).uniform_(0, 1)
x10 = torch.diag(torch.ones(3))

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())  # boolean True/False
print(tensor.short())  # int16
print(tensor.long())  # int64 (Important)
print(tensor.half())  # float16
print(tensor.float())  # float32 (Important)
print(tensor.double())  # float64

# Array to Tensor conversion and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
