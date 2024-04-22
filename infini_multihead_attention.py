import torch
from torch import nn

class InfiniMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_k, dim_v, dim_input, segment_len, dropout=0.1):
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_input = dim_input
        self.num_heads = num_heads
        self.segment_len = segment_len
        self.dropout = dropout

        self.proj_k = nn.Linear(dim_input, num_heads * dim_k, bias=False)
        self.proj_v = nn.Linear(dim_input, num_heads * dim_v, bias=False)
        self.proj_q = nn.Linear(dim_input, num_heads * dim_k, bias=False)
        self.proj_out = nn.Linear(num_heads * dim_v, dim_input, bias=False)
      
        self.mask=torch.triu(torch.ones(segment_len, segment_len), diagonal=1).to("mps")
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, dim_v))
        self.dropout_layer = nn.Dropout(self.dropout)

        # self.memory:[batch_size, num_heads, dim_k, dim_v]
        self.memory=torch.zeros(1, num_heads, dim_k, dim_v).to("mps")

        # self.z:[batch_size, num_heads, 1, dim_k]
        self.z=(torch.randn(1, num_heads, 1, dim_k)*1e-10).to("mps")

    
    def forward(self, x):
        #x:[batch_size, seq_len, dim_input]
        batch_size, seq_len, _ = x.shape
        n_seq = seq_len // self.segment_len 
        out = []
        for i in range(n_seq):
            #x_segment:[batch_size, segment_len, dim_input]
            x_segment = x[:, i*self.segment_len:(i+1)*self.segment_len, :]
            
            #k,v,q:[batch_size, num_heads, segment_len, dim_k/dim_v]
            k=self.proj_k(x_segment).unsqueeze(1).view(batch_size, self.num_heads, self.segment_len, self.dim_k)
            q=self.proj_q(x_segment).unsqueeze(1).view(batch_size, self.num_heads, self.segment_len, self.dim_k)
            v=self.proj_v(x_segment).unsqueeze(1).view(batch_size, self.num_heads, self.segment_len, self.dim_v)

            #att_dot:[batch_size, num_heads, segment_len, dim_v]
            att_dot=self._att_dot(q, k, v)

            #att_mem:[batch_size, num_heads, segment_len, dim_v]
            att_mem=self._memory_retrival(batch_size, q)

            self._memory_update(k, v)

            #att:[batch_size, num_heads, segment_len, dim_v]
            att=nn.functional.sigmoid(self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * att_dot
            att = att.view((batch_size, self.segment_len, self.num_heads * self.dim_v))
            att=self.proj_out(att)
            att=self.dropout_layer(att)
            out.append(att)

        #out:[batch_size, seq_len, dim_input]
        return torch.cat(out, dim=1)



    def _att_dot(self, q, k, v):
        #k,v,q:[batch_size, num_heads, segment_len, dim_k/dim_v]
        scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_k))
        #scores:[batch_size, num_heads, segment_len, segment_len]
        scores =scores.masked_fill(self.mask.bool(), float('-inf'))
        scores=nn.functional.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        att_dot = scores @ v
        return att_dot
    
    def _memory_retrival(self, batch_size, q):
        # if self.memory==None and self.z==None:
        #     self.memory=torch.zeros(batch_size, self.num_heads, self.dim_k, self.dim_v)
        #     self.z=torch.zeros(batch_size, self.num_heads, 1, self.dim_k)

        sigma_q = (nn.functional.elu(q) + 1.0)
        #sigma_q:[batch_size, num_heads, segment_len, dim_k]
        #self.memory:[batch_size, num_heads, dim_k, dim_v]
        #self.z:[batch_size, num_heads, 1, dim_k]
        #att_mem:[batch_size, num_heads, segment_len, dim_v]
        att_mem = ((sigma_q @ self.memory) / (sigma_q @ self.z.transpose(-2, -1))).detach()
        return att_mem
    
    def _memory_update(self, k, v):
        #k,v:[batch_size, num_heads, segment_len, dim_k/dim_v]
        sigma_k = nn.functional.elu(k) + 1.0
        # self.memory:[batch_size, num_heads, dim_k, dim_v]
        if self.memory!=None:
            self.memory = self.memory + sigma_k.transpose(-2, -1) @ v
        else:
            self.memory = sigma_k.transpose(-2, -1) @ v
        
        # self.z:[batch_size, num_heads, 1, dim_k]
        if self.z!=None:
            self.z = self.z + sigma_k.sum(dim=-2, keepdim=True)
        else:
            self.z = sigma_k.sum(dim=-2, keepdim=True)

def test():  
    torch.set_default_device("mps") 
    dim_input = 128
    num_heads = 8
    dim_k = dim_input//num_heads
    dim_v = dim_input//num_heads
    segment_len=2

    model = InfiniMultiHeadAttention(num_heads, dim_k, dim_v, dim_input, segment_len)
    batch = torch.randn(2, 8, dim_input)
    print(batch[0])

    print(model(batch)[0])


if __name__ == "__main__":
    test() 
    

    
