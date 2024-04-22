import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_k, dim_v, dim_input, dropout=0.1):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_input = dim_input
        self.num_heads = num_heads
        self.dropout = dropout

        self.proj_k = nn.Linear(dim_input, num_heads * dim_k, bias=False)
        self.proj_v = nn.Linear(dim_input, num_heads * dim_v, bias=False)
        self.proj_q = nn.Linear(dim_input, num_heads * dim_k, bias=False)
        self.proj_out = nn.Linear(num_heads * dim_v, dim_input, bias=False)
      
        self.dropout_layer = nn.Dropout(self.dropout)

    
    def forward(self, x):
       #x:[batch_size, seq_len, dim_input]
        batch_size, seq_len, _ = x.shape
        self.mask=torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to("mps")
        
        #k,v,q:[batch_size, num_heads, seq_len, dim_k/dim_v]
        k=self.proj_k(x).unsqueeze(1).view(batch_size, self.num_heads, seq_len, self.dim_k)
        q=self.proj_q(x).unsqueeze(1).view(batch_size, self.num_heads, seq_len, self.dim_k)
        v=self.proj_v(x).unsqueeze(1).view(batch_size, self.num_heads, seq_len, self.dim_v)

        #att_dot:[batch_size, num_heads, seq_len, dim_v]
        att_dot=self._att_dot(q, k, v)

        att = att_dot.view((batch_size, seq_len, self.num_heads * self.dim_v))
        att=self.proj_out(att)
        out=self.dropout_layer(att)

        #out:[batch_size, seq_len, dim_input]
        return out



    def _att_dot(self, q, k, v):
        #k,v,q:[batch_size, num_heads, seq_len, dim_k/dim_v]
        scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_k))
        #scores:[batch_size, num_heads, seq_len, seq_len]
        scores =scores.masked_fill(self.mask.bool(), float('-inf'))
        scores=nn.functional.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        att_dot = scores @ v
        return att_dot
    

def test():   
    dim_input = 128
    num_heads = 8
    dim_k = dim_input//num_heads
    dim_v = dim_input//num_heads

    model = MultiHeadAttention(num_heads, dim_k, dim_v, dim_input)
    batch = torch.randn(2, 4, dim_input)

    print(model(batch).shape)


if __name__ == "__main__":
    test() 
    

    
