from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim_input, dropout=0.1):
        super().__init__()
        self.dim_input = dim_input
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.dim_input, out_features=self.dim_input * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_input * 4, out_features=self.dim_input),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)