import torch
import torch.nn as nn
from signlib import FEAT_DIM
import config as C


class SignNet(nn.Module):
    """Two-stream late-fusion network.
    Stream 1: skeletal keyframes -> BiGRU.   Stream 2: RGB keyframe embeddings -> BiGRU."""
    def __init__(self, n_classes, hid=128, use_rgb=True, p=0.3):
        super().__init__()
        self.use_rgb, self.hid = use_rgb, hid
        self.s_in = nn.Sequential(nn.Linear(FEAT_DIM, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(p))
        self.s_gru = nn.GRU(hid, hid, 2, batch_first=True, bidirectional=True, dropout=p)
        dim = 4 * hid                                   # mean + max pooled, both directions
        if use_rgb:
            self.r_in = nn.Sequential(nn.Dropout(0.5), nn.Linear(C.RGB_DIM, hid), nn.LayerNorm(hid), nn.GELU())
            self.r_gru = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
            dim += hid
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(dim, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, n_classes))

    def forward(self, skel, rgb):
        h, _ = self.s_gru(self.s_in(skel))
        z = torch.cat([h.mean(1), h.max(1).values], -1)
        if self.use_rgb:
            r, _ = self.r_gru(self.r_in(rgb))
            z = torch.cat([z, r.mean(1)], -1)
        return self.head(z)
