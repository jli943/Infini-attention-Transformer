from torch import nn
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward
from infini_multihead_attention import InfiniMultiHeadAttention

class TransformerBlock(nn.Module):

    def __init__(self, num_heads, dim_input, dropout=0.1, infini=False, segment_len=3):
        super().__init__()
        self.dim_input = dim_input
        self.num_heads = num_heads
        self.dropout = dropout
        if not infini:
            self.multi_head_attention_layer = MultiHeadAttention(num_heads=num_heads, dim_k=(dim_input//num_heads), dim_v=(dim_input//num_heads), dim_input=dim_input, dropout=dropout)
        else:
            self.multi_head_attention_layer = InfiniMultiHeadAttention(num_heads=num_heads, dim_k=(dim_input//num_heads), dim_v=(dim_input//num_heads), dim_input=dim_input, segment_len=segment_len, dropout=dropout)

        self.feed_forward_layer = FeedForward(dim_input=dim_input, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.dim_input)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.dim_input)

    def forward(self, x):
        # Note: The order of the operations is different from the original Transformer paper
        # The order here is: LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # Residual connection
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x