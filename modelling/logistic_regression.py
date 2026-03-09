import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=vocab_size, out_features=1, bias=True) # y = xA^T + b
        self.sigmoid = nn.Sigmoid()

    # (batch_size, vocab_size) -> (batch_size, 1)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return self.sigmoid(y).squeeze(1)