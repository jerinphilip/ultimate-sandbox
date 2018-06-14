import torch
import os

params = {
    "dtype" : torch.float,
    "device" : torch.device("cpu")
}


input_size, output_size = 1000, 10
hidden_size = 100
batch_size = 64

x = torch.randn(batch_size, input_size, **params)
z = torch.randn(batch_size, output_size, **params)


W1 = torch.randn(input_size, hidden_size, **params)
W2 = torch.randn(hidden_size, output_size, **params)

learning_rate = 1e-6

for epoch in range(500):
    h = x.mm(W1)
    h_relu = h.clamp(min=0)
    y = h_relu.mm(W2)

    # MSE Loss
    loss = (y - z).pow(2).sum().item()

    dy = 2.0 * (y - z)
    dW2 = h_relu.t().mm(dy)
    dh_relu = dy.mm(W2.t())
    dh = dh_relu.clone()

    # Piecewise differentiability
    dh[ h < 0 ] = 0
    dW1 = x.t().mm(dh)

    # Update Weights
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    print(epoch, loss)


