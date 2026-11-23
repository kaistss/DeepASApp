# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding

from src.models.obj_path.obj_attn.vanilla.obj_path import ObjectBlock
from src.models.tf_locoformer.utils.sub_decoder import DoANet, ClassNet
from src.models.tf_locoformer.locoformer_separator import TFLocoformerBlock

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class TFLocoformerSeparator(nn.Module):
    """TF-Locoformer model presented in [1].

    Reference:
    [1] Kohei Saijo, Gordon Wichern, François G. Germain, Zexu Pan, and Jonathan Le Roux,
    "TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation
    and Enhancement," in Proc. International Workshop on Acoustic Signal Enhancement (IWAENC),
    Sep. 2024.

    Args:
        num_spk: int
            number of output sources/speakers.
        n_layers: int
            number of Locoformer blocks.
        emb_dim: int
            Size of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
            Feed-forward network (FFN)-type chosen from "conv1d" or "swiglu_conv1d".
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimensions in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel: int
            Kernel size in Conv1d.
        conv1d_shift: int
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        eps: float
            Small constant for normalization layer.
    """

    def __init__(
        self,
        n_fft: int = 640,
        hop_length: int = 320,
        n_mics: int = 4,
        num_spk: int = 2,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgrouporm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        pos_enc: str = "rope",
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
        ckpt: str = None
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mics = n_mics
        self.num_spk = num_spk
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2*n_mics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),  # gLN
        )

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        if pos_enc == "nope":
            pe_freq = pe_time = None
        elif pos_enc == "rope":
            pe_freq = RotaryEmbedding(attention_dim // n_heads)
            pe_time = RotaryEmbedding(attention_dim // n_heads)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    pe_freq,
                    pe_time,
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.objconv = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim*num_spk, ks, padding=padding),
            nn.Mish(inplace=True),
            nn.GroupNorm(1, emb_dim*num_spk)
        )

        self.aneconv_pop = nn.ConvTranspose2d(emb_dim, n_mics * 2, ks, padding=padding)
        self.revconv_pop = nn.ConvTranspose2d(emb_dim, n_mics * 2, ks, padding=padding)
        self.nseconv = nn.ConvTranspose2d(emb_dim, n_mics * 2, ks, padding=padding)

        self.clsdecoder_pop = ClassNet(input_channels=emb_dim, n_cnn_filters=64)

        self.doadecoder_pop = DoANet(input_channels=emb_dim, n_cnn_filters=64)

        self.conv_pop = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim*num_spk, ks, padding=padding),
            nn.Mish(inplace=True),
            nn.GroupNorm(1, emb_dim*num_spk, eps=eps)
        )

        self.pop = nn.ModuleList([])
        for idx in range(n_layers//2, n_layers):
            self.pop.append(ObjectBlock(emb_dim, num_spk, dropout, t_ksize))

        self.down_conv_pop_3 = nn.Conv2d(emb_dim, 2*n_mics, ks, padding=padding)
        self.down_conv_pop_4 = nn.Conv2d(emb_dim, 2*n_mics, ks, padding=padding)

        self.clsdecoder_pop_3 = ClassNet(input_channels=emb_dim, n_cnn_filters=emb_dim)
        self.clsdecoder_pop_4 = ClassNet(input_channels=emb_dim, n_cnn_filters=emb_dim)

        self.doadecoder_pop_3 = DoANet(input_channels=emb_dim, n_cnn_filters=emb_dim)
        self.doadecoder_pop_4 = DoANet(input_channels=emb_dim, n_cnn_filters=emb_dim)

        if ckpt is not None:
            self._initialize_parameter(ckpt)

        self._freeze_parameter(['pop'])


    def _initialize_parameter(self, ckpt_path):
        cont = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        new_cont = {}
        for k, v in cont['state_dict'].items():
            if 'model.' in k:
                new_key = k.replace('model.', '')
            if new_key in self.state_dict().keys() and self.state_dict()[new_key].shape == v.shape:
                new_cont[new_key] = v
        missing_keys, unexpected_keys = self.load_state_dict(new_cont, strict=False)
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Missing keys: {missing_keys}")


    def _freeze_parameter(self, trainable_keys):
        train_keys = []
        for name, param in self.named_parameters():
            if any([trainable_key in name for trainable_key in trainable_keys]):
                param.requires_grad = True
                train_keys.append(name)
            else:
                param.requires_grad = False
        print(f'Trainable keys: {train_keys}')


    def forward(self, input_dict: dict) -> torch.Tensor:
        """Forward.

        Args:
            input (torch.Tensor): batched single-channel audio tensor with
                in TF-domain [B, T, F]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len num_spk
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in the output.
        """

        input = input_dict['mixture']
        B, C, L = input.shape

        input = torch.stft(
            input.reshape(-1, L),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(input.device),
            return_complex=True
        )
        input = input.reshape(B, C, *input.shape[1:]).transpose(-1, -2)  # [B, M, T, F]

        label_indices = input_dict.get('label_indices', None)
        if label_indices is not None: label_indices = label_indices.reshape(B*self.num_spk, -1)

        # if input.ndim == 3:
        #     # in case the input does not have channel dimension
        #     batch0 = input.unsqueeze(1)
        # elif input.ndim == 4:
        #     assert input.shape[1] == 1, "Only monaural input is supported."
        #     batch0 = input.transpose(1, 2)  # [B, M, T, F]

        batch = torch.cat((input.real, input.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        with torch.amp.autocast("cuda", enabled=False):
            batch = self.conv(batch)  # [B, -1, T, F]

        # separation
        for ii in range(self.n_layers):
            if ii == self.n_layers // 2:
                batch_pop = self.conv_pop(batch)
                batch_pop = batch_pop.reshape(n_batch, self.num_spk, self.emb_dim, n_frames, n_freqs)
            with torch.no_grad():
                batch = self.blocks[ii](batch.detach())  # [B, -1, T, F]
            if ii >= self.n_layers // 2:
                batch_pop = self.pop[ii - self.n_layers // 2](batch, batch_pop)
            if ii == 3:
                batch_pop_3 = batch_pop.reshape(-1, self.emb_dim, n_frames, n_freqs)
                with torch.amp.autocast("cuda", enabled=False):
                    sep_out_3_ = self.down_conv_pop_3(batch_pop_3)
                    cls_out_3, sed_out_3, onoff_out_3 = self.clsdecoder_pop_3(batch_pop_3.transpose(2, 3))
                    doa_out_3 = self.doadecoder_pop_3(batch_pop_3.transpose(2, 3))
                cls_out_3 = cls_out_3.reshape(B, self.num_spk, *cls_out_3.shape[1:])
                sed_out_3 = sed_out_3.reshape(B, self.num_spk, *sed_out_3.shape[1:])
                onoff_out_3 = onoff_out_3.reshape(B, self.num_spk, *onoff_out_3.shape[1:])
                doa_out_3 = doa_out_3.reshape(B, self.num_spk, *doa_out_3.shape[1:])
            if ii == 4:
                batch_pop_4 = batch_pop.reshape(-1, self.emb_dim, n_frames, n_freqs)
                with torch.amp.autocast("cuda", enabled=False):
                    sep_out_4_ = self.down_conv_pop_4(batch_pop_4)
                    cls_out_4, sed_out_4, onoff_out_4 = self.clsdecoder_pop_4(batch_pop_4.transpose(2, 3))
                    doa_out_4 = self.doadecoder_pop_4(batch_pop_4.transpose(2, 3))
                cls_out_4 = cls_out_4.reshape(B, self.num_spk, *cls_out_4.shape[1:])
                sed_out_4 = sed_out_4.reshape(B, self.num_spk, *sed_out_4.shape[1:])
                onoff_out_4 = onoff_out_4.reshape(B, self.num_spk, *onoff_out_4.shape[1:])
                doa_out_4 = doa_out_4.reshape(B, self.num_spk, *doa_out_4.shape[1:])

        objbatch = batch_pop.view([B, self.num_spk, self.emb_dim, n_frames, n_freqs]).view([-1, self.emb_dim, n_frames, n_freqs])

        with torch.amp.autocast("cuda", enabled=False):
            aneout_ = self.aneconv_pop(objbatch)
            revout_ = self.revconv_pop(objbatch)
            nseout_ = self.nseconv(batch)

        sep_out_3_ = sep_out_3_.reshape(n_batch, self.num_spk, 2, self.n_mics, n_frames, n_freqs)
        sep_out_3_ = sep_out_3_.permute(0, 1, 3, 5, 4, 2).contiguous()
        sep_out_3_ = torch.view_as_complex(sep_out_3_)
        sep_out_3  = torch.zeros(n_batch * self.num_spk, self.n_mics, L, device=sep_out_3_.device)
        for m in range(self.n_mics):
            sep_out_channel = sep_out_3_[:, :, m].view([n_batch * self.num_spk, n_freqs, n_frames])
            sep_out_3[:, m] = torch.istft(
                sep_out_channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(sep_out_3.device)
            )
        sep_out_3  = sep_out_3.view([n_batch, self.num_spk, self.n_mics, L])

        sep_out_4_ = sep_out_4_.reshape(n_batch, self.num_spk, 2, self.n_mics, n_frames, n_freqs)
        sep_out_4_ = sep_out_4_.permute(0, 1, 3, 5, 4, 2).contiguous()
        sep_out_4_ = torch.view_as_complex(sep_out_4_)
        sep_out_4  = torch.zeros(n_batch * self.num_spk, self.n_mics, L, device=sep_out_4_.device)
        for m in range(self.n_mics):
            sep_out_channel = sep_out_4_[:, :, m].view([n_batch * self.num_spk, n_freqs, n_frames])
            sep_out_4[:, m] = torch.istft(
                sep_out_channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(sep_out_4.device)
            )
        sep_out_4  = sep_out_4.view([n_batch, self.num_spk, self.n_mics, L])        

        aneout_ = aneout_.view([B, self.num_spk, 2, self.n_mics, n_frames, n_freqs])
        aneout_ = aneout_.permute(0, 1, 3, 5, 4, 2).contiguous()
        aneout_ = torch.view_as_complex(aneout_)
        aneout  = torch.zeros(B * self.num_spk, self.n_mics, L, device=aneout_.device)
        for m in range(self.n_mics):
            aneout_channel = aneout_[:, :, m].view([B * self.num_spk, n_freqs, n_frames])
            aneout[:, m] = torch.istft(
                aneout_channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(aneout.device)
            )
        aneout = aneout.view([B, self.num_spk, self.n_mics, L])

        revout_ = revout_.view([B, self.num_spk, 2, self.n_mics, n_frames, n_freqs])
        revout_ = revout_.permute(0, 1, 3, 5, 4, 2).contiguous()
        revout_ = torch.view_as_complex(revout_)
        revout  = torch.zeros(B * self.num_spk, self.n_mics, L, device=revout_.device)
        for m in range(self.n_mics):
            revout_channel = revout_[:, :, m].view([B * self.num_spk, n_freqs, n_frames])
            revout[:, m] = torch.istft(
                revout_channel,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(revout.device)
            )
        revout = revout.view([B, self.num_spk, self.n_mics, L])

        nseout_ = nseout_.view([B, 2, self.n_mics, n_frames, n_freqs])
        nseout_ = nseout_.permute(0, 2, 4, 3, 1).contiguous()
        nseout_ = torch.view_as_complex(nseout_)
        nseout  = torch.zeros(B, self.n_mics, L, device=nseout_.device)
        for m in range(self.n_mics):
            nseout[:, m] = torch.istft(
                nseout_[:, m],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(nseout.device)
            )

        cls_out, sed_out, onoff_out = self.clsdecoder_pop(objbatch.transpose(2, 3)) # , label_indices=label_indices)
        cls_out = cls_out.view([B, self.num_spk, *cls_out.shape[1:]])
        sed_out = sed_out.view([B, self.num_spk, *sed_out.shape[1:]])
        onoff_out = onoff_out.reshape(B, self.num_spk, *onoff_out.shape[1:])

        doa_out = self.doadecoder_pop(objbatch.transpose(2, 3))
        doa_out = doa_out.view([B, self.num_spk, *doa_out.shape[1:]])

        # with torch.amp.autocast("cuda", enabled=False):
        #     batch = self.deconv(batch)  # [B, num_spk*2, T, F]
        # batch = batch.view([n_batch, self.num_spk, 2, n_frames, n_freqs])

        # batch = torch.complex(batch[:, :, 0], batch[:, :, 1])

        out = {
            'anechoic': aneout,
            'reverb': revout,
            'noise': nseout,
            'cls_label': cls_out,
            'sed_label': sed_out,
            'doa_label': doa_out,
            'onoff_label': onoff_out,
            'sep_out_3': sep_out_3,
            'sep_out_4': sep_out_4,
            'cls_out_3': cls_out_3,
            'cls_out_4': cls_out_4,
            'sed_out_3': sed_out_3,
            'sed_out_4': sed_out_4,
            'onoff_out_3': onoff_out_3,
            'onoff_out_4': onoff_out_4,
            'doa_out_3': doa_out_3,
            'doa_out_4': doa_out_4
        }

        return out
