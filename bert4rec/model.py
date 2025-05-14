import torch.nn as nn
import torch

class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=128, max_len=100, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.item_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(self.norm(x))
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        return self.output(x)