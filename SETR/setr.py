"""
Segmentation Transformer for semantic segmentation.
"""
import torch
import torch.nn as nn
from transformer import TransformerEncoder
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
        transformer_forward_expansion=4,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
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
        self.transformer_forward_expansion = transformer_forward_expansion
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation


        # Image to sequence settings
        self.num_patches = (img_size // patch_size) ** 2
        self.seq_length = self.num_patches
        self.patch_flatten_dim = patch_size * patch_size * num_channels


        # Whether use convolution or a single linear to transform a 
        # patch to a sequence
        if self.conv_patch_representation:
            self.conv_patch = nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=embedding_dim,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                padding=self._get_padding(
                    'VALID', (self.patch_size, self.patch_size)
                ),
            )
            self.patch_linear_projection = None
        else:
            self.conv_patch = None
            self.patch_linear_projection = nn.Linear(self.patch_flatten_dim, self.embedding_dim)


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


        # Transformer setup
        self.transformer = TransformerEncoder(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.dropout_rate,
            self.attn_dropout_rate,
            self.transformer_forward_expansion
        )


    def _init_decoder(self):
        raise NotImplementedError("Decoder should be implemented in child class!")


    def encode(self, x):
        # Input batch image: [N, num_channels, height, width]
        N, C, H, W = x.shape

        # Convert the image to a patch sequence
        if self.conv_patch_representation:
            # Use convolution to convert to sequecne
            x = self.conv_patch(x)                          # [N, embedding_dim, patch_size, patch_size]
            x = x.permute(0, 2, 3, 1).contiguous()          # [N, patch_size, patch_size, embedding_dim]
            x = x.view(x.size(0), -1, self.embedding_dim)  # [N, seq_length, embedding_dim]
            
        else:
            # Use a linaer projection to convert to sequence 
            x = (
                x.unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
                .contiguous()
            )   # [N, num_channels, height_num_patch, width_num_patch, patch_size, patch_size]
            x = x.view(N, C, -1, self.patch_size ** 2)          # [N, num_channels, num_patches, patch_area]
            x = x.permute(0, 2, 3, 1).contiguous()              # [N, num_patches, patch_area, num_channels]
            x = x.view(x.size(0), -1, self.patch_flatten_dim)   # [N, num_patches(seq_len), patch_flatter_dim]
            x = self.patch_linear_projection(x)

        # positional encoding
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # transformer
        x, intmd_x = self.transformer(x, return_intermediate=True)

        return x, intmd_x


    def decode(self, x):
        raise NotImplementedError("Decode function should be implemented in child class!")
    

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                auxillary_outputs[i] = intmd_encoder_outputs[i]
            return decoder_output, auxillary_outputs
        
        return decoder_output


    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)
    

    def _reshape_output(self, x):
        # Reshape the sequence back to a 2d image.
        # x: [N, seq_len(patch_num), embedding_dim]
        # img: [N, embedding_dim, height_patch_num, width_patch_num]
        x = x.view(
            x.size(0),
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.embedding_dim
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x





class SETR_Naive(SETR):
    def __init__(
        self,
        img_size,
        patch_size,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        transformer_foward_expansion,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            transformer_forward_expansion=transformer_foward_expansion,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type
        )

        self.num_classes = num_classes
        self._init_decoder()

    
    def _init_decoder(self):
        self.img_project = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                kernel_size=1,
                stride=1,
                padding=self._get_padding('VALID', (1, 1))
            ),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.embedding_dim,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=self._get_padding('VALID', (1, 1))
            )
        )
        self.upsample = nn.Upsample(
            scale_factor=self.patch_size, mode='bilinear', align_corners=False
        )


    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.img_project(x)
        x = self.upsample(x)
        return x






if __name__ == "__main__":
    img_size = 128
    patch_size = 16
    N = 4
    num_channels = 3
    num_classes = 7
    flatten_dim = patch_size**2 * num_channels
    embedding_dim = 256
    num_heads = 8
    num_layers = 24
    transformer_forward_expansion = 4
    dropout_rate = 0.1
    attn_dropout_rate = 0.1
    conv_patch_representation = False
    positional_encoding_type = "learned"


    img = torch.randn((N, num_channels, img_size, img_size))

    model = SETR_Naive(
        img_size=img_size,
        patch_size=patch_size,
        num_channels=num_channels,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        transformer_foward_expansion=transformer_forward_expansion,
        dropout_rate=dropout_rate,
        attn_dropout_rate=attn_dropout_rate,
        conv_patch_representation=conv_patch_representation,
        positional_encoding_type=positional_encoding_type
    )

    out = model(img)
    assert out.shape == torch.Size([N, num_classes, img_size, img_size])
    print("img shape:", img.shape)
    print("output shape:", out.shape)

