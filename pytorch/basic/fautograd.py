
import torch

class relu_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, y):
        y, = ctx.saved_tensors
        cx = y.clone()
        cx[y < 0] = 0
        return cx


kw = {
    "dtype": torch.float,
    "device": torch.device("cuda:0")
}

batch_size, input_size, hidden_size, output_size = 64, 1000, 100, 10

X = torch.randn(batch_size, input_size, **kw)
Z = torch.randn(batch_size, output_size, **kw)


W1 = torch.randn(input_size, hidden_size, requires_grad=True, **kw)
W2 = torch.randn(hidden_size, output_size, requires_grad=True, **kw)

learning_rate = 1e-6
for epoch in range(500):
    relu = relu_.apply
    Y = relu(X.mm(W1)).mm(W2)
    loss = (Y-Z).pow(2).sum()
    print(epoch, loss.item())
    loss.backward()
    with torch.no_grad():
        W1 -= learning_rate*W1.grad
        W2 -= learning_rate*W2.grad

        W1.grad.zero_()
        W2.grad.zero_()



