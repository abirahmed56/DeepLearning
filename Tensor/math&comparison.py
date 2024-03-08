import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)

z = x + y
print(z)
# Subtraction
z = x - y
print(z)

# Division
z = torch.true_divide(x, y)
print(z)

# inplace operation
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x is not inplace

# Exponentiation
z = x.pow(2)
print(z)
z = x ** 2
print(z)
# Simple comparison
z = x > 0
print(z)
z = x < 0
print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # 2 * 3
print(x3)
x3 = x1.mm(x2)  # are same

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))

# element wise mult.
print(f"x = {x}, y ={y}")
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand(batch, n, m)
tensor2 = torch.rand(batch, m, p)
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
print(z)
z = x1 ** x2
print(z)
# other tensor operation
sum_x = torch.sum(x, dim=0)  # x.max(dim=0)
values, indices = torch.max(x, dim=0)
print(f"max value of x: {values}, indices: {indices}")
values, indices = torch.min(x, dim=0)
print(f"min values of x:{values}, indices: {indices}")
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
print(z)
z = torch.argmin(x, dim=0)
print(z)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)  # equal
print(z)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)  # any value less than zero set zero and more than 10 will set 10
print(z)
x = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)
