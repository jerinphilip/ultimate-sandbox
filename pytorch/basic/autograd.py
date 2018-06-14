import torch
import os

torch.manual_seed(7)
params = {
    "dtype" : torch.float,
    "device" : torch.device("cpu")
}


input_size, output_size = 1000, 10
hidden_size = 100
batch_size = 64

x = torch.randn(batch_size, input_size, **params)
z = torch.randn(batch_size, output_size, **params)


W1 = torch.randn(input_size, hidden_size, requires_grad=True, **params)
W2 = torch.randn(hidden_size, output_size, requires_grad=True, **params)

learning_rate = 1e-6

for epoch in range(500):
    y = x.mm(W1).clamp(min=0).mm(W2)

    # MSE Loss
    loss = (y - z).pow(2).sum()
    print(epoch, loss.item())

    # Replacing own differentiation
    # dy = 2.0 * (y - z)
    # dW2 = h_relu.t().mm(dy)
    # dh_relu = dy.mm(W2.t())
    # dh = dh_relu.clone()
    # Piecewise differentiability
    # Taken care of
    # dh[ h < 0 ] = 0
    # dW1 = x.t().mm(dh)

    # With autograd, yay!
    # Sets x.grad on each variables.
    loss.backward()


    # Update Weights
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        W2 -= learning_rate * W2.grad

        # Manually set grads to zero, for next iteration.
        # Questions: Are there cases when I don't set grad to zero?
        W1.grad.zero_()
        W2.grad.zero_()

