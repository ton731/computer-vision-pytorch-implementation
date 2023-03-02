import torch
import torch.nn as nn



class FixedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()

        position = torch.arange(max_seq_length).unsqueeze(1)    # [max_seq_length, 1]
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # [1, max_seq_length, embedding_dim]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [N, seq_length, embedding_dim]
        N, seq_length, _ = x.shape
        position_embeddings = self.pe[:, :seq_length, :].expand(N, -1, -1)
        return x + position_embeddings



class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()

        self.pe = nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer(
            "positions",
            torch.arange(max_seq_length).expand((1, -1)),
        )

    def forward(self, x):
        # x: [N, seq_length, embedding_dim]
        N, seq_length, _ = x.shape
        positions = self.positions[:, :seq_length].expand(N, seq_length)
        position_embeddings = self.pe(positions)
        return x + position_embeddings


