"""
Segmentation Transformer for semantic segmentation.
"""
import torch
import torch.nn as nn
from transformer import Transformer
from positional_encoding import LearnedPositionalEncoding, FixedPositionalEncoding


class SETR(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            conv_patch_representation=False,
            positional_encoding_type="learned",
    ):
        super().__init__()

        assert img_size % patch_size == 0
        assert embedding_dim % num_heads == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # Image to sequence settings
        self.num_patches = (img_size // patch_size) ** 2
        self.seq_length = self.num_patches
        self.patch_flatten_dim = patch_size * patch_size * num_channels

        # Whether use convolution or a single linear to transform a 
        # patch to a sequence
        if self.conv_patch_representation:
            pass
        else:
            self.conv_patch = None
            self.patch_encoding = nn.Linear(self.patch_flatten_dim, self.embedding_dim)

        # Position embedding settings
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.seq_length, self.embedding_dim
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)




