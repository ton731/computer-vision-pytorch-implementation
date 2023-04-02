import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, attn_dropout_rate):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(self.head_dim * heads, embed_size)  # head_dim * heads = embed_size
        self.dropout = nn.Dropout(attn_dropout_rate)

    def forward(self, x):
        # Get number of training examples, x: [N, q_len, embed_size]
        N, seq_len, _ = x.shape

        # Get the sequence lens, k_len should be equal to v_len
        v_len, k_len, q_len = seq_len, seq_len, seq_len

        v = self.values(x)  # [N, v_len, embed_size]
        k = self.keys(x)    # [N, k_len, embed_size]
        q = self.queries(x) # [N, q_len, embed_size]

        # Split the embedding into self.heads different pieces
        v = v.reshape(N, v_len, self.heads, self.head_dim)
        k = k.reshape(N, k_len, self.heads, self.head_dim)
        q = q.reshape(N, q_len, self.heads, self.head_dim)

        # Calculate the score (energy) for each key when given a query,
        # use Einsum to do the matrix multiplication for Q x K
        # q shape: [N, q_len, heads, head_dim]
        # k shape: [N, k_len, heads, head_dim]
        # energy: [N, heads, q_len, k_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        # Normalize energy values so it becomes attention weights in the
        # k_len dimension, so k_len dim sums equals to 1. Also divide by
        # scaling factor for better stability
        # attention shape: [N, heads, q_len, k_len]
        attention = torch.softmax((energy / (self.embed_size ** (1/2))), dim=3)

        # Multiply the attention weight with the values, and combine the result 
        # from different heads
        # attention shape: [N, heads, q_len, k_len]
        # v shape: [N, v_len, heads, head_dim]
        # out after matrix multiply: [N, q_len, heads, head_dim], then
        # we reshape and flattern the last two dimension
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            N, q_len, self.heads * self.head_dim
        )

        # Final output shape: [N, q, embed_size]
        out = self.fc_out(out)
        out = self.dropout(out)

        return out



class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_size, 
            heads,
            dropout_rate,
            attn_dropout_rate,
            forward_expansion
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.self_attention = SelfAttention(embed_size, heads, attn_dropout_rate)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Layer Norm --> Self Attention --> Layer Norm --> MLP
        attention = self.self_attention(self.norm1(x))
        x = attention + x
        out = self.feed_forward(self.norm2(x))
        out = out + x
        return out



class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        forward_expansion=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size=embed_size,
                heads=heads,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
                forward_expansion=forward_expansion
            ) for _ in range(num_layers)]
        )

    def forward(self, x, return_intermediate=True):
        intermiediate_outputs = []
        for layer in self.layers:
            x = layer(x)
            intermiediate_outputs.append(x)
        return x, intermiediate_outputs if return_intermediate else x
    





if __name__ == "__main__":

    N = 8
    seq_len = 256
    embedding_dim = 256
    input_sequence = torch.randn((N, seq_len, embedding_dim))

    model = TransformerEncoder(
        embed_size=embedding_dim,
        num_layers=8,
        heads=8,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        forward_expansion=4
    )
    print("model:")
    print(model)
    
    out = model(input_sequence)
    assert out.shape == torch.Size([N, seq_len, embedding_dim])
    print(out.shape)
