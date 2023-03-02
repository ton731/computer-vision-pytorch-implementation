import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(self.head_dim * heads, embed_size)  # head_dim * heads = embed_size

    def forward(self, v, k, q, mask):
        # Get number of training examples, q: [N, q_len, embed_size]
        N = q.shape[0]

        # Get the sequence lens, k_len should be equal to v_len
        v_len, k_len, q_len = v.shape[1], k.shape[1], q.shape[1]

        v = self.values(v)  # [N, v_len, embed_size]
        k = self.keys(k)    # [N, k_len, embed_size]
        q = self.queries(q) # [N, q_len, embed_size]

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

        # Mask padded indicies so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

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

        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.self_attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out



class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()

        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )

        # In the Encoder, the value, key, and query are all the same, but it'll change in the Decoder.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out



class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()

        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # src_mask is for not computing th gradients of the padded parts in the sequence
        # tgt_mask is for the masking the right-side inputs in the decoder
        attention = self.self_attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out



class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super().__init__()

        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
        
        out = self.fc_out(x)

        return out



class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            tgt_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # [N, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out


