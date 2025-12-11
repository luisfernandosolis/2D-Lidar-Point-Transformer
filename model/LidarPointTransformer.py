import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarPointTransformer(nn.Module):
    """
    Minimal Lidar point transformer:
    - Input: B x N x C  (e.g., C = [x,y,optional features])
    - Output: B x N x num_classes (per-point classification)
    """
    def __init__(self, in_dim=2, embed_dim=128, n_heads=4, n_layers=2, num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2, dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: B x N x C
        mask: B x N : True where valid (optional)
        """
        B, N, C = x.shape
        x_proj = self.input_proj(x)  # B x N x H
        # Transformer expects (S, N, E) or (seq_len, batch_size, embed)
        x_trans = x_proj.permute(1, 0, 2)  # N, B, E
        if mask is not None:
            # Transformer uses key_padding_mask where True indicates padding position to be ignored
            # mask: B x N -> convert to boolean where True means ignore
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        out = self.transformer_encoder(x_trans, src_key_padding_mask=key_padding_mask)  # N, B, E
        out = out.permute(1, 0, 2)  # B, N, E
        logits = self.mlp_head(out)  # B x N x num_classes
        return logits

if __name__ == "__main__":
    # rudimentary sanity check
    model = LidarPointTransformer(in_dim=2, embed_dim=64, n_heads=4, n_layers=2, num_classes=4)
    x = torch.rand(2, 100, 2)
    logits = model(x)
    print("Logits shape:", logits.shape)  # expected (2,100,4)
