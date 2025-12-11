import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Standard in Llama 2 for improved training stability.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalization based on root mean square
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    The activation function used in Llama 2 and PaLM.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False) # Value
        self.w3 = nn.Linear(hidden_dim, dim, bias=False) # Output

    def forward(self, x):
        # x = (xW1 * SiLU) * xW2 -> W3
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# --- 2. Transformer Encoder Block ---

class LlamaEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Normalization for Attention (Residual connection)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Pre-Normalization for Feed-Forward (Residual connection)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x

# --- 3. Main Model: LidarPointTransformer ---

class LidarPointTransformer(nn.Module):
    def __init__(self, 
                 input_dim=2,          # x, y (2D LiDAR)
                 model_dim=128,        # Internal Transformer dimension
                 num_heads=4,          # Attention heads
                 num_layers=4,         # Depth of the model
                 ffn_dim=512,          # Hidden dimension of SwiGLU
                 output_dim=512,       # Final dimension (Must match CLIP/Text)
                 dropout=0.1):
        super().__init__()

        # 1. Input Embedding (Point -> Vector)
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # 2. Positional Encoding (Learnable)
        # We assume a maximum number of points per cluster (e.g., 256)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, model_dim)) 

        # 3. Stack of Encoder Blocks (Llama 2 style)
        self.layers = nn.ModuleList([
            LlamaEncoderBlock(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final Normalization
        self.norm_final = RMSNorm(model_dim)
        
        # 5. Projection Head (For contrastive space)
        self.projector = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, output_dim)
        )

    def forward(self, x):
        """
        x shape: (Batch_Size, Num_Points, 2)
        """
        B, N, C = x.shape
        
        # Initial Embedding
        x = self.embedding(x) # (B, N, model_dim)
        
        # Add Positional Encoding (truncated to current number of points)
        # Note: Ideally use Rotary Embeddings (RoPE) for production, 
        # but absolute learnable embeddings are sufficient for a start.
        x = x + self.pos_embedding[:, :N, :]
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_final(x)
        
        # --- Pooling (Aggregation) ---
        # Convert sequence of N points into 1 single vector per cluster.
        # Max Pooling is highly effective for Point Clouds (PointNet logic)
        x_global = torch.max(x, dim=1)[0] # (B, model_dim)
        
        # --- Final Projection ---
        output_vector = self.projector(x_global) # (B, output_dim)
        
        # Normalize the vector (Crucial for Cosine Similarity in Contrastive Loss)
        output_vector = F.normalize(output_vector, p=2, dim=1)
        
        return output_vector