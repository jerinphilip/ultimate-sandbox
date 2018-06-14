import numpy as np
from tqdm import tqdm

input_size, output_size = 1000, 10
hidden_size = 100
batch_size = 64

x = np.random.randn(batch_size, input_size)
z = np.random.randn(batch_size, output_size)


W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

learning_rate = 1e-6

for epoch in range(500):
    h = x.dot(W1)
    h_relu = np.maximum(h, 0)
    y = h_relu.dot(W2)

    # MSE Loss
    loss = np.square(y - z).sum()

    dy = 2.0 * (y - z)
    dW2 = h_relu.T.dot(dy)
    dh_relu = dy.dot(W2.T)
    dh = dh_relu.copy()

    # Piecewise differentiability
    dh[ h < 0 ] = 0
    dW1 = x.T.dot(dh)

    # Update Weights
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    print(epoch, loss)


