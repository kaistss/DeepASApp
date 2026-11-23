import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ObjectBlock(nn.Module):
    """
    Generic feature → Object feature 단방향 주입.
    Object 분리가 목적이므로 O2G(역방향) 제거.
    """
    def __init__(self, input_dim, n_srcs, dropout, kernel_sizs):
        super().__init__()
        self.n_srcs = n_srcs

        ks, padding = (kernel_sizs, 3), (kernel_sizs // 2, 1)

        self.obj_conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim*n_srcs, ks, padding=padding),
            nn.Mish(inplace=True),
            nn.GroupNorm(1, input_dim*self.n_srcs)
        )

        self.ffn1_dos  = GatedConv(input_dim * 2, input_dim, c_ks=(7, 7), g_ks=(7, 7))
        self.ffn1_obj  = GatedConv(input_dim * 2, input_dim, c_ks=(7, 7), g_ks=(7, 7))
        self.ffn1_fuse = GatedConv(input_dim * 2, input_dim, c_ks=(7, 7), g_ks=(7, 7))

        self.spk_attn  = SpeakerModule(input_dim, heads=4, n_src=self.n_srcs, dropout=dropout)


    def forward(self, feature_pre, feature_dos):
        """
        feature_pre: [B, D, T, F]
        feature_obj: [B, S, D, T, F]
        반환:        [B, S, D, T, F]
        """
        B, D, T, Fq = feature_pre.shape
        S = self.n_srcs

        feature_dos = feature_dos.reshape(-1, D, T, Fq)

        # Object Splitting
        feature_obj = self.obj_conv(feature_pre).reshape(-1, D, T, Fq) # [B*S, D, T, F]

        feature_inp  = torch.cat([feature_dos, feature_obj], dim=1)
        feature_dos_ = self.ffn1_dos(feature_inp) + feature_dos
        feature_obj_ = self.ffn1_obj(feature_inp) + feature_obj
        feature_dos  = self.ffn1_fuse(torch.cat([feature_obj_, feature_dos_], dim=1)) + feature_dos # [B*S, D, T, F]

        feature_dos = feature_dos.reshape(B, S, D, T, Fq)
        feature_dos  = self.spk_attn(feature_dos)

        return feature_dos


class SpeakerModule(nn.Module):
    def __init__(self, dim, heads, n_src, dropout):
        super().__init__()
        self.n_src = n_src

        self.attn = SpeakerAttention(dim, dim // heads, heads=heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, S, D, T, Fq = x.shape

        x = x.permute(0, 3, 4, 1, 2).reshape(-1, self.n_src, D) # [BTF, S, D]

        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x

        x = x.reshape(-1, T, Fq, self.n_src, D).permute(0, 3, 4, 1, 2).contiguous()

        return x


class SpeakerAttention(nn.Module):

    def __init__(self, dim, inner_dim, heads, dropout):
        super().__init__()

        self.heads = heads
        self.scale = inner_dim ** -0.5

        self.to_q = nn.Linear(dim, inner_dim * heads, bias=False)
        self.to_k = nn.Linear(dim, inner_dim * heads, bias=False)
        self.to_v = nn.Linear(dim, inner_dim * heads, bias=False)

        self.dropout = dropout

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * heads, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Note: this attention runs over the (tiny) n_srcs axis, but the leading
        # batch dim gets folded together with T and F upstream (see DeepASA.forward),
        # which can reach into the hundreds of thousands. flash_attn's CUDA kernel
        # launches a grid indexed by batch*heads and is capped at 65535, so it
        # throws "invalid configuration argument" here. Plain SDPA has no such cap
        # and is plenty cheap for a sequence length of n_srcs.
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads) # [B, S, D]
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, c_ks=(3, 3), g_ks=(7, 7)):
        super().__init__()

        c_pd = (c_ks[0] // 2, c_ks[1] // 2)
        g_pd = (g_ks[0] // 2, g_ks[1] // 2)

        self.norm = nn.GroupNorm(2, in_channels)

        self.c_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.c_dw = nn.Conv2d(out_channels, out_channels, kernel_size=c_ks, padding=c_pd, groups=out_channels)
        self.c_norm = nn.GroupNorm(1, out_channels)

        self.g_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.g_dw = nn.Conv2d(out_channels, out_channels, kernel_size=g_ks, padding=g_pd, groups=out_channels)
        self.g_norm = nn.GroupNorm(1, out_channels)

        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        content = self.c_dw(self.swish(self.c_norm(self.c_pw(x))))
        gate    = self.g_dw(self.swish(self.g_norm(self.g_pw(x))))
        return content * self.sigmoid(gate)
