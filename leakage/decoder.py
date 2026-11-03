"""Decoder of the scene-leakage measurement (Table 4): one embedding vector -> one RGB frame in [-1, 1].

Linear(D -> 512*4*4), then 5 up-blocks (4 -> 8 -> 16 -> 32 -> 64 -> 128), each ConvTranspose2d(stride 2) + GroupNorm + GELU
+ Conv3x3 + GroupNorm + GELU, then Conv3x3 -> Tanh. The same architecture is used for every representation; only D changes.
"""
import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, n_groups=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(n_groups, c_out), c_out)
        self.conv = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, c_out), c_out)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm(self.up(x)))
        x = self.act(self.norm2(self.conv(x)))
        return x


class FrameDecoder(nn.Module):
    def __init__(self, in_dim, out_res=128, base_ch=64):
        super().__init__()
        assert out_res == 128, "the architecture produces 128x128 frames"
        self.in_dim = in_dim
        self.out_res = out_res

        c0 = base_ch * 8  # 512
        self.fc = nn.Sequential(
            nn.Linear(in_dim, c0 * 4 * 4),
        )
        self.start_shape = (c0, 4, 4)

        self.up1 = UpBlock(c0,           base_ch * 8)   # 4  -> 8
        self.up2 = UpBlock(base_ch * 8,  base_ch * 4)   # 8  -> 16
        self.up3 = UpBlock(base_ch * 4,  base_ch * 2)   # 16 -> 32
        self.up4 = UpBlock(base_ch * 2,  base_ch)       # 32 -> 64
        self.up5 = UpBlock(base_ch,      base_ch)       # 64 -> 128

        self.out = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, emb):
        x = self.fc(emb).view(emb.size(0), *self.start_shape)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.out(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
