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

loss_fn = nn.MSELoss(size_average=False)

X = torch.randn(batch_size, input_size, **kw)
Z = torch.randn(batch_size, output_size, **kw)

learning_rate = 1e-6
for epoch in range(500):
    Y = model(X)
    loss = loss_fn(Y, Z)
    print(epoch, loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
