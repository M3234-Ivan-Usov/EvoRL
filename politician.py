import torch
from torch.nn.functional import one_hot, sigmoid, softmax


class Politician(torch.nn.Module):
    def __init__(self, n_metrics, n_arms):
        super().__init__()
        self.n_metrics, self.n_arms = n_metrics, n_arms
        self.lin = torch.nn.Linear(n_metrics, n_arms)
        self.opt = torch.optim.RMSprop(self.parameters())

    def forward(self, metrics):
        x = torch.from_numpy(metrics).float()
        x = self.lin(x)
        x = sigmoid(x)
        return softmax(x)

    def backprop(self, policies, arms, advantages):
        used_arms = one_hot(arms, num_classes=self.n_arms)
        loss = -(torch.log(policies) * used_arms * advantages).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

