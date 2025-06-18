import torch
from torch_geometric.graphgym.models.gnn import GNNPreMP
from .rrwp_pe import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder
from .grit_layer import GritTransformerLayer
import torch.nn as nn


class TripleChannelNodeEncoder(nn.Module):
    """Handling and fusing basic features in Elliptic++ dataset using three different channel"""

    def __init__(self, emb_dim, drop_rate=0.3):
        super().__init__()

        # Local Features (93)
        self.local_proj = nn.Sequential(
            nn.Linear(93, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(drop_rate * 0.2),
            nn.Linear(128, emb_dim // 2),  # emb_dim/2
        )

        # Aggregated (72)
        self.agg_proj = nn.Sequential(
            nn.Linear(72, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, emb_dim // 4),  # emb_dim/4
        )

        # New Features in E++ dataset (17)
        self.dirty_proj = nn.Sequential(
            nn.Linear(17, 8),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(drop_rate),
            nn.Linear(8, emb_dim // 8),  # emb_dim/8
        )

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim // 2 + emb_dim // 4 + emb_dim // 8, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x):
        # Feature Segmentation
        x_local = x[:, :93]  # Local Features
        x_agg = x[:, 93 : 93 + 72]  # Aggregated Features
        x_dirty = x[:, 93 + 72 :]  # New Features in E++ dataset

        # Independent Encoding of Three Channels
        h_local = self.local_proj(x_local)  # [B, emb_dim//2]
        h_agg = self.agg_proj(x_agg)  # [B, emb_dim//4]
        h_dirty = self.dirty_proj(x_dirty)  # [B, emb_dim//8]

        # Fusion
        fused = torch.cat([h_local, h_agg, h_dirty], dim=1)
        return self.fusion(fused)  # [B, emb_dim]


class DIY_NodeHead(torch.nn.Module):
    """Presenting finial prediction results"""

    def __init__(self, dim_in=64, dim_out=2, hidden_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ]
            dim_in = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.mlp = nn.Sequential(*layers)

    def _apply_index(self, batch):
        mask = "{}_mask".format(batch.split[0])
        return batch.x[batch[mask]], batch.binary_label[batch[mask]]

    def forward(self, batch):
        batch.x = self.mlp(batch.x)
        pred, label = self._apply_index(batch)

        mask = "{}_mask".format(batch.split[0])
        hidden_state = batch.x[batch[mask]]
        return pred, label, hidden_state


class FeatureEncoder(torch.nn.Module):
    """Encoding node and edge features"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_encoder = TripleChannelNodeEncoder(hidden_size)

    def forward(self, batch):
        batch.x = self.node_encoder(batch.x)
        # batch.edge_attr = self.edge_encoder(batch.edge_attr)
        return batch


class GritTransformer(torch.nn.Module):
    """The proposed GritTransformer (Graph Inductive Bias Transformer)"""

    def __init__(
        self,
        dim_out,
        hidden_size=96,
        ksteps=17,
        layers_pre_mp=0,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
        attn_dropout=0.5,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_size)
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_size)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(
            ksteps,
            hidden_size,
            pad_to_full_graph=True,
            add_node_attr_as_self_loop=False,
            fill_value=0.0,
        )

        self.layers_pre_mp = layers_pre_mp
        if layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(hidden_size, hidden_size, layers_pre_mp)

        layers = [
            GritTransformerLayer(
                in_dim=hidden_size,
                out_dim=hidden_size,
                num_heads=n_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        ]
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = DIY_NodeHead(
            dim_in=hidden_size, dim_out=dim_out, dropout=dropout
        )

    def forward(self, batch):
        batch = self.get_embd(batch)
        return self.post_mp(batch)

    def get_embd(self, batch):
        batch = self.encoder(batch)
        batch = self.rrwp_abs_encoder(batch)
        batch = self.rrwp_rel_encoder(batch)
        if self.layers_pre_mp > 0:
            batch = self.pre_mp(batch)

        return self.layers(batch)
