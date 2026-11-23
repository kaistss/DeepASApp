# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter


class DoANet(nn.Module):
    def __init__(self, time_dim=201, freq_dim=321, input_channels=64, pool_size=[[8,5],[8,1]], pool_time=True,  n_cnn_filters=64, rnn_size=128, n_rnn=2,fc_size=128, dropout_perc=0., verbose=False):
        super(DoANet, self).__init__()
        self.verbose = verbose
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        self.freq_pooled_size = int(freq_dim / np.prod(np.array(pool_size), axis=0)[0])
        #building CNN feature extractor
        conv_layers = []
        in_chans = input_channels
        for p in pool_size:
            curr_chans = n_cnn_filters
            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    nn.Mish(inplace=True),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
        self.cnn = nn.Sequential(*conv_layers)
        self.rnn = nn.GRU(n_cnn_filters * self.freq_pooled_size, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc) # cnn output channel(freq) 만큼 수정 26.07.28
        self.doa = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, 3),
                    nn.Tanh())
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,3,1,2) #[batch, time, channels, freq]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.rnn(x)
        doa = self.doa(x)
        return doa # , self.cnn[0][0].weight


class ClassNet(nn.Module):
    def __init__(self, time_dim=201, freq_dim=321, input_channels=64, n_cls=13, pool_size_t=[[2,5],[2,1],[2,1],[2,1],[2,1],[2,1],[1,1]], pool_size_f=[[8,5],[4,4]], pool_time=True, n_cnn_filters=64, rnn_size=128, n_rnn=2, fc_size=128, dropout_perc=0., use_arcface=True, verbose=False):
        super(ClassNet, self).__init__()
        self.use_arcface = use_arcface
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size_t), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        self.freq_pooled_size_t = int(freq_dim / np.prod(np.array(pool_size_t), axis=0)[0])
            
        # building T-CNN feature extractor
        conv_layers_t = []
        in_chans = input_channels
        for i, p in enumerate(pool_size_t):
            curr_chans = n_cnn_filters
            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers_t.append(
                nn.Sequential(
                    nn.Conv2d(in_chans if i == 0 else in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  # padding 1 = same with kernel = 3
                    nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    # nn.Mish(inplace=True),
                    nn.GELU(),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
        self.cnn_t = nn.Sequential(*conv_layers_t)
        #building F-CNN feature extractor
        conv_layers_f = []
        in_chans = input_channels
        for p in pool_size_f:
            curr_chans = n_cnn_filters
            pool = [p[0],p[1]]
            conv_layers_f.append(
                nn.Sequential(
                    nn.Conv2d(in_chans if i == 0 else in_chans, out_channels=curr_chans, kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    FrequencyWiseNorm(freq_dim),
                    # nn.GroupNorm(1, n_cnn_filters, 1e-5),
                    nn.Mish(inplace=True),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans
            freq_dim = freq_dim // p[0]
        self.cnn = nn.Sequential(*conv_layers_f)
        # SED
        self.Linear_t = nn.Sequential(
            # nn.Linear(320 + 768, 540),
            nn.Linear(n_cnn_filters * self.freq_pooled_size_t, 540), # cnn output channel(freq) 만큼 수정 26.07.28
            nn.LayerNorm(540),
            nn.GELU(),
        )
        self.Linear_c = nn.Sequential(
            nn.Conv2d(n_cnn_filters, 128, kernel_size=1), # cnn output channel 만큼 수정 26.07.28
            FrequencyWiseNorm(freq_dim),
            nn.Mish(inplace=True),
        )
        self.Linear_f = nn.Sequential(
            nn.Linear(128, 540),
            nn.LayerNorm(540),
            nn.Mish(inplace=True),
        )
        self.time_gru = nn.GRU(540, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc)
        self.freq_gru = nn.GRU(540, rnn_size, num_layers=n_rnn, batch_first=True, bidirectional=True, dropout=dropout_perc)
        self.t_asp = AttentiveStatsPool(2 * rnn_size)
        self.f_asp = AttentiveStatsPool(2 * rnn_size)
        # self.norm = nn.LayerNorm(2 * 2 * rnn_size)
        if self.use_arcface:
            self.Linear = ArcMarginProduct(2 * 2 * rnn_size, n_cls)
        else:
            self.Linear = nn.Linear(2 * 2 * rnn_size, n_cls)

        self.onoff = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, 1))
        self.sed = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, n_cls))

    def forward(self, feature, label_indices=None):
        input = feature # [B, C, F, T]

        # CNN path
        t_feature = self.cnn_t(input)
        t_feature = t_feature.permute(0,3,1,2) #[batch, time, channels, freq]
        t_feature = t_feature.reshape(t_feature.shape[0], t_feature.shape[1], -1)
        
        f_feature = self.cnn(input)     # [batch, channels, freq, time]
        f_feature = self.Linear_c(f_feature).mean(dim=-1).permute(0, 2, 1).contiguous() # [batch, freq, time, channel]

        t_feature = self.time_gru(self.Linear_t(t_feature))[0]
        f_feature = self.freq_gru(self.Linear_f(f_feature))[0]

        onoff = self.onoff(t_feature).squeeze(dim=-1)
        sed = self.sed(t_feature)

        t_feature = self.t_asp(t_feature)
        f_feature = self.f_asp(f_feature)

        t_feature = F.normalize(t_feature, dim=-1)
        f_feature = F.normalize(f_feature, dim=-1)

        embedding = torch.cat([t_feature, f_feature], dim=-1)

        if self.use_arcface:
            cls, logit = self.Linear(embedding, label_indices)
        else:
            cls = self.Linear(embedding)

        # return output
        return cls, sed, onoff


class FrequencyWiseNorm(nn.Module):
    """주파수 축별(채널·시간 독립) 평균/분산 정규화"""
    def __init__(self, freq_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, freq_dim, 1))
        self.beta  = nn.Parameter(torch.zeros(1, 1, freq_dim, 1))
        self.eps = eps

    def forward(self, x):
        # x: (N,C,F,T)  →  평균·분산을 F축(마지막) 단위로 계산
        mean = x.mean(dim=(1, 3), keepdim=True)
        var  = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features=128,
        out_features=200,
        s=32.0,
        m=0.50,
        sub=1,
        easy_margin=False,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0, 1.0)
        
        if label is not None:
            # Handle outliers (label = -1)
            valid_mask = (label != -1).reshape(-1)
            if not valid_mask.any():
                return cosine * self.s  # Return regular cosine similarity for all samples if no valid labels
                
            # Process only valid samples
            valid_x = x[valid_mask]
            valid_label = label[valid_mask]
            valid_cosine = cosine[valid_mask]
            
            if self.sub > 1:
                valid_cosine = valid_cosine.view(-1, self.out_features, self.sub)
                valid_cosine, _ = torch.max(valid_cosine, dim=2)
                
            sine = torch.sqrt(1.0 - torch.pow(valid_cosine, 2))
            phi = valid_cosine * self.cos_m - sine * self.sin_m
            
            if self.easy_margin:
                phi = torch.where(valid_cosine > 0, phi, valid_cosine)
            else:
                phi = torch.where((valid_cosine - self.th) > 0, phi, valid_cosine - self.mm)

            one_hot = torch.zeros(valid_cosine.size(), device=x.device)
            one_hot.scatter_(1, valid_label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * valid_cosine)
            output = output * self.s
            
            # Create final output with original shape
            
            final_output = (cosine * self.s).float()  # Initialize with regular cosine similarity
            final_output[valid_mask] = output  # Replace valid samples with ArcFace output
            
            return final_output, F.linear(x, self.weight)
        else:
            return cosine * self.s, F.linear(x, self.weight)


class AttentiveStatsPool(nn.Module):
    """
    x의 shape: (batch, frames, dim)
    output shape: (batch, 2*dim)
      - [가중 평균(mu), 가중 표준편차(sigma)] 연결(concat)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(AttentiveStatsPool, self).__init__()
        # 작은 Attention 모듈
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear3 = nn.Linear(2*input_dim, input_dim)
        
    def forward(self, x):
        e = self.linear2(torch.tanh(self.linear1(x)))  # (B, T, 1)
        alpha = F.softmax(e, dim=1)  # (B, T, 1)
        mu = torch.sum(alpha * x, dim=1)  # (B, D)
        second_moment = torch.sum(alpha * x * x, dim=1)  # (B, D)
        sigma = torch.sqrt(torch.clamp(second_moment - mu * mu, min=1e-5))  # (B, D)
        pooled = torch.cat([mu, sigma], dim=1)  # (B, 2D)
        return self.linear3(pooled)