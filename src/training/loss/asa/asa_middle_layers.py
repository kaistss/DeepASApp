from torch import Tensor
import torch
from torch import nn

from torch.nn import functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr
from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio as sa_sdr
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate as permutate
from torch.nn.functional import cross_entropy as ce
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from collections import Counter
import math
from torch.nn import Parameter


from typing import *

def neg_sa_sdr(preds: Tensor, target: Tensor, scale_invariant: bool = True) -> Tensor:
    batch_size, _, channels, _ = target.shape
    sa_sdr_val = sum([sa_sdr(preds=preds[:, :, m], target=target[:, :, m], scale_invariant=scale_invariant) for m in range(channels)])
    return -torch.sum(sa_sdr_val.view(batch_size, -1), dim=1)

def audioclassification_loss(output, target, silence_target_prob=1/18, label_smoothing=0.0, keep_batch=False):
    B, C = output.size()
    loss = [] if keep_batch else 0.
    for batch_idx in range(B):
        # target이 0 ~ 12인 경우 일반 Cross-Entropy Loss 사용
        if target[batch_idx] != -1:
            loss_ = torch.nn.functional.cross_entropy(output[batch_idx].unsqueeze(0), target[batch_idx].unsqueeze(0), label_smoothing=label_smoothing)
        else:
            # target이 -1인 경우 Silence Loss 사용
            silence_output = output[batch_idx]
            silence_target = torch.full_like(silence_output, silence_target_prob)
            loss_ = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(silence_output, dim=0), silence_target, reduction='batchmean')
        if keep_batch:
            loss.append(loss_)
        else:
            loss = loss + loss_
    loss = torch.stack(loss, dim=0) if keep_batch else loss / B
    return loss


class AudioClassificationLoss(nn.Module):
    def __init__(self, num_classes=18, label_smoothing=0.0, keep_batch=False):
        super(AudioClassificationLoss, self).__init__()
        self.silence_target_prob = 1/num_classes
        self.label_smoothing = label_smoothing
        self.keep_batch = keep_batch

    def forward(self, output, target):
        """
        output: (batch_size, num_classes) - Model's predicted probabilities/logits.
        target: (batch_size,) - Ground truth labels, with -1 indicating silence (OOD).
        """
        return audioclassification_loss(output, target, self.silence_target_prob, self.label_smoothing, self.keep_batch)


class Loss(nn.Module):
    is_scale_invariant_loss: bool
    name: str
    def __init__(self, loss_func: Callable, pit: bool, loss_func_kwargs: Dict[str, Any] = dict()):
        super().__init__()

        self.loss_func = loss_func
        self.pit = pit
        self.loss_func_kwargs = loss_func_kwargs
        self.is_scale_invariant_loss = {
            neg_sa_sdr: True if 'scale_invariant' in loss_func_kwargs and loss_func_kwargs['scale_invariant'] == True else False,
        }[loss_func]
        self.name = loss_func.__name__

    def forward(self, preds: Tensor, target: Tensor, reorder: bool = None, reduce_batch: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        perms = None
        if self.pit:
            # print(f'preds-pit: {preds.shape}, target-pit: {target.shape}')
            losses, perms = pit(preds=preds, target=target, metric_func=self.loss_func, eval_func='min', mode='permutation-wise', **self.loss_func_kwargs)
            if reorder:
                preds = permutate(preds, perm=perms)
        else:
            losses = self.loss_func(preds=preds, target=target, **self.loss_func_kwargs)

        return losses.mean() if reduce_batch else losses, perms, preds

    def extra_repr(self) -> str:
        kwargs = ""
        for k, v in self.loss_func_kwargs.items():
            kwargs += f'{k}={v},'

        return f"loss_func={self.loss_func.__name__}({kwargs}), pit={self.pit}"


def get_loss_func():
    def loss_func(output, target):
        B, S, M, L = output['anechoic'].shape
        neg_sa_sdr_loss = Loss(loss_func=neg_sa_sdr, pit=True, loss_func_kwargs={'scale_invariant': True})
        cls_loss = AudioClassificationLoss(num_classes=13, label_smoothing=0.0, keep_batch=False)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        # Separation loss
        sep_out_, sep_label = output['anechoic'] + output['reverb'], target['anechoic'] + target['reverb']
        sep_loss_val, perms, sep_out = neg_sa_sdr_loss(preds=sep_out_, target=sep_label, reorder=True, reduce_batch=True)
        
        # Anechoic loss
        ane_out, ane_label = permutate(output['anechoic'], perm=perms), target['anechoic']
        ane_loss_val = neg_sa_sdr(preds=ane_out, target=ane_label).mean()
        
        # Reverb loss
        rev_out, rev_label = permutate(output['reverb'], perm=perms), target['reverb']
        rev_loss_val = neg_sa_sdr(preds=rev_out, target=rev_label).mean()

        # noise loss
        nse_snr_val = -0.01 * snr(output['noise'], target['noise']).mean()
        # classification loss
        cls_out = permutate(output['cls_label'], perm=perms)
        cls_loss_val = sum([cls_loss(cls_out[:, i], target['cls_label'][:, i]) for i in range(S)]) / S
        # SED loss
        sed_out = permutate(output['sed_label'], perm=perms)
        sed_loss_val = mse_loss(sed_out, target['sed_label'])
        # DoA loss
        doa_out = permutate(output['doa_label'], perm=perms)
        doa_loss_val = mse_loss(doa_out, target['doa_label'])
        # On-Off loss
        onoff_out = permutate(output['onoff_label'], perm=perms)
        onoff_loss_val = bce_loss(onoff_out, target['onoff_label'])

        # Sep loss for layer 1
        # sep_out_3 = permutate(output['ane_out_3'] + output['rev_out_3'], perm=perms)
        sep_out_3 = permutate(output['sep_out_3'], perm=perms)
        sep_loss_3_val = neg_sa_sdr(preds=sep_out_3, target=sep_label).mean()

        sep_out_4 = permutate(output['sep_out_4'], perm=perms)
        sep_loss_4_val = neg_sa_sdr(preds=sep_out_4, target=sep_label).mean()

        cls_out_3 = permutate(output['cls_out_3'], perm=perms)
        cls_loss_3_val = sum([cls_loss(cls_out_3[:, i], target['cls_label'][:, i]) for i in range(S)]) / S

        cls_out_4 = permutate(output['cls_out_4'], perm=perms)
        cls_loss_4_val = sum([cls_loss(cls_out_4[:, i], target['cls_label'][:, i]) for i in range(S)]) / S

        sed_out_3 = permutate(output['sed_out_3'], perm=perms)
        sed_loss_3_val = mse_loss(sed_out_3, target['sed_label'])

        sed_out_4 = permutate(output['sed_out_4'], perm=perms)
        sed_loss_4_val = mse_loss(sed_out_4, target['sed_label'])

        onoff_out_3 = permutate(output['onoff_out_3'], perm=perms)
        onoff_loss_3_val = bce_loss(onoff_out_3, target['onoff_label'])

        onoff_out_4 = permutate(output['onoff_out_4'], perm=perms)
        onoff_loss_4_val = bce_loss(onoff_out_4, target['onoff_label'])

        doa_out_3 = permutate(output['doa_out_3'], perm=perms)
        doa_loss_3_val = mse_loss(doa_out_3, target['doa_label'])

        doa_out_4 = permutate(output['doa_out_4'], perm=perms)
        doa_loss_4_val = mse_loss(doa_out_4, target['doa_label'])

        # Total loss
        loss = sep_loss_val + ane_loss_val + rev_loss_val + nse_snr_val + cls_loss_val + sed_loss_val + doa_loss_val + onoff_loss_val \
            + 0.1 * (sep_loss_4_val + cls_loss_4_val + sed_loss_4_val + onoff_loss_4_val + doa_loss_4_val) \
            + 0.1 * (sep_loss_3_val + cls_loss_3_val + sed_loss_3_val + onoff_loss_3_val + doa_loss_3_val)        
            # + 0.1 * (sep_loss_4_val + doa_loss_4_val) \
            # + 0.1 * (sep_loss_3_val + doa_loss_3_val)
            
        loss_dict = {
            'loss': loss,
            'sep_loss': sep_loss_val,
            'ane_loss': ane_loss_val,
            'rev_loss': rev_loss_val,
            'nse_snr': nse_snr_val,
            'cls_loss': cls_loss_val,
            'sed_loss': sed_loss_val,
            'doa_loss': doa_loss_val,
            'onoff_loss': onoff_loss_val,
            'sep_loss_3': sep_loss_3_val,
            'sep_loss_4': sep_loss_4_val,
            'cls_loss_3': cls_loss_3_val,
            'cls_loss_4': cls_loss_4_val,
            'sed_loss_3': sed_loss_3_val,
            'sed_loss_4': sed_loss_4_val,
            'onoff_loss_3': onoff_loss_3_val,
            'onoff_loss_4': onoff_loss_4_val,
            'doa_loss_3': doa_loss_3_val,
            'doa_loss_4': doa_loss_4_val
        }
        perm_out = {
            'sep_out': sep_out.detach(),
            'anechoic': ane_out.detach(),
            'reverb': rev_out.detach(),
            'cls_label': cls_out.detach(),
            'sed_label': sed_out.detach(),
            'doa_label': doa_out.detach(),
            'onoff_label': onoff_out.detach()
        }
        return loss_dict, perm_out
    return loss_func
