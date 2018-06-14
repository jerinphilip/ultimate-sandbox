import torch
import torch.nn as nn

kw = {
    "dtype": torch.float,
    "device": torch.device("cuda:0")
}

batch_size, input_size, hidden_size, output_size = 64, 1000, 100, 10

model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size))


model = model.cuda()

learning_rate = 1e-6
loss_fn = nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

X = torch.randn(batch_size, input_size, **kw)
Z = torch.randn(batch_size, output_size, **kw)

for epoch in range(5000):
    Y = model(X)
    loss = loss_fn(Y, Z)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

