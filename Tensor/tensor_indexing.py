import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x[0].shape)  # x[0, : ]

print(x[:, 0].shape)
print(x[2, 0:10])  # 0:10 --> [0, 1, ..., 9]

x[0, 0] = 100

# fancy indexing
x = torch.arange(18)
indices = [2, 5, 9]
print(x[indices])

x = torch.rand((3, 5))
print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operation
print(torch.where(x > 5, x, x * 2))
print(torch.tensor([0, 0, 1, 1, 2, 2]).unique())
print(x.ndimension())  # 5x5x5->3
print(x.numel())
