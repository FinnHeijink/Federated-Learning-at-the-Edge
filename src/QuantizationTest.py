import torch
import torch.nn as nn

a = torch.Tensor([1, 2, 3, 4]).to(torch.float16)
b = torch.Tensor([4, 3, 2, 251]).to(torch.float16)
print(a+b)

input = torch.Tensor([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]]).to(torch.float32)
targets = torch.Tensor([
    [0],
    [0],
    [0],
    [1]
]).to(torch.float32)

class GradientReversalModule(torch.autograd.Function):
    def __init__(self, lambd):
        super(GradientReversalModule,self).__init__()
        self.lambd = lambd

    def forward(self,x):
        return x
    def backward(self,grad_value):
        return -grad_value*self.lambd

layerWeights = torch.empty((1, 2))
layerParameter = torch.nn.Parameter(layerWeights)
criterion = nn.MSELoss()

optimizer = torch.optim.SGD([layerParameter], lr=0.001)

for i in range(1000):
    optimizer.zero_grad()
    prediction = torch.nn.functional.linear(input, layerWeights)
    loss = criterion(prediction, targets)
    loss.backward()
    optimizer.step()

    if i % 100==0:
        print(loss.item())

