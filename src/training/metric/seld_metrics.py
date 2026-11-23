import torch
import torch.nn.functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr


def _si_sdri(est, label, mixture, eps=1e-8):
    """
    Scale-invariant SDR improvement over the (unseparated) mixture baseline.
    est, label, mixture: [B, S, M, L]
    Zero-padded source slots (all-zero label) are excluded from the average.
    Returns: [B]
    """
    active_mask = label.pow(2).sum(dim=(-2, -1)) > eps   # [B, S]
    sisdr_est = si_sdr(est, label)        # [B, S, M]
    sisdr_mix = si_sdr(mixture, label)    # [B, S, M]
    sisdri = sisdr_est - sisdr_mix        # [B, S, M]

    active_mask = active_mask.unsqueeze(-1).expand_as(sisdri)   # [B, S, M]
    sisdri = torch.where(active_mask, sisdri, torch.zeros_like(sisdri))
    denom = active_mask.sum(dim=(1, 2)).clamp(min=1)
    return sisdri.sum(dim=(1, 2)) / denom   # [B]


def SELD_metrics(output, target):
    """
    Calculate Sound Event Localization and Detection (SELD) metrics
    Parameters:
    - sed_out: Tensor of shape [B, S, T, C] with sound event detection output
    - sed_label: Tensor of shape [B, S, T, C] with sound event detection ground truth
    - doa_out: Tensor of shape [B, S, T, 3] with DOA estimation output
    - doa_label: Tensor of shape [B, S, T, 3] with DOA estimation ground truth
    Returns:
    - ER: Error rate of the sound event detection (Spatial Error Rate)
    - F1: F1 score of the sound event detection (Spatial F1 Score)
    - LE: Localization error of the DOA estimation
    - LR: Localization recall of the DOA estimation
    """

    # 0. Separation metric
    ane_out = output['anechoic']
    rev_out = output['reverb']
    sep_out = ane_out + rev_out

    ane_label = target['anechoic'].to(ane_out.device)
    rev_label = target['reverb'].to(rev_out.device)
    noise_label = target['noise'].to(ane_out.device)
    sep_label = ane_label + rev_label

    # mixture baseline (the unseparated input fed to the model), broadcast over sources
    mixture = (ane_label.sum(dim=1) + rev_label.sum(dim=1) + noise_label).unsqueeze(1).expand_as(sep_label)

    ane_sisdri = _si_sdri(ane_out, ane_label, mixture)
    rev_sisdri = _si_sdri(rev_out, rev_label, mixture)
    sep_sisdri = _si_sdri(sep_out, sep_label, mixture)

    # 1) 각 source, class별로 TP, TN, FP, FN 계산
    sed_out = output['sed_label']
    doa_out = output['doa_label']
    sed_label = target['sed_label']
    doa_label = target['doa_label']

    sed_label = sed_label.to(sed_out.device)
    doa_label = doa_label.to(doa_out.device)
    TP = (sed_out * sed_label)   # True Positives                                        # [B, S, T, C]
    TN = ((1 - sed_out) * (1 - sed_label))   # True Negatives                            # [B, S, T, C]
    FP = (sed_out * (1 - sed_label))     # False Positives                               # [B, S, T, C]
    FN = ((1 - sed_out) * sed_label)     # False Negatives                               # [B, S, T, C]
    # 2) 각 source, class별로 LE, LR 계산
    active_mask = torch.norm(doa_label, dim=-1) > 0.5   # Active sound event mask                   # [B, S, T]
    cos_sim = torch.sum(F.normalize(doa_out, dim=-1) * F.normalize(doa_label, dim=-1), dim=-1)      # [B, S, T]
    angles = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))                                            # [B, S, T]
    LE = torch.sum(angles * active_mask.float()) / (torch.sum(active_mask) + 1e-8)   # Localization Error      # [B]
    LR = TP.sum() / (TP.sum() + FN.sum() + 1e-8)  # Localization Recall                                               # [B]
    # 3) 각 source, class별로 TP_{c<=20\deg} 계산
    within_threshold = (angles <= 20).unsqueeze(3).float().to(sed_label.device) * sed_label  # Localization Recall within threshold         # [B, S, T, C]
    TP_wi_th = TP * within_threshold    # True Positives within threshold   # [B, S, T, C]
    FP_wi_th = FP + (TP - TP_wi_th)     # False Positives within threshold  # [B, S, T, C]
    # 4) 각 source, class별로 ER, F1 계산
    ER = torch.max(FP_wi_th.sum(), FN.sum()) / (TP.sum() + FN.sum() + 1e-8)  # Error Rate (Spatial Error Rate)
    # ER = (torch.min(FP_wi_th.sum(), FN.sum()) + torch.max(torch.tensor(0, device=FN.device), FP_wi_th.sum() - FN.sum()) + torch.max(torch.tensor(0, device=FN.device), FN.sum() - FP_wi_th.sum())) / (TP.sum() + FN.sum() + 1e-8)
    F1 = 2 * TP_wi_th.sum() / (2 * TP.sum() + FP.sum() + FN.sum() + 1e-8)  # F1 Score (Spatial F1 Score)
    seld_score = (ER + (1 - F1) + LE / 180 + (1 - LR)) / 4
    out = {
        'ER': 100*ER,
        'F1': 100*F1,
        'LE': LE,
        'LR': 100*LR,
        'seld_score': seld_score,
        'ane_sisdri': ane_sisdri,
        'rev_sisdri': rev_sisdri,
        'sep_sisdri': sep_sisdri
    }
    return out


def SELD_metrics_c(sed_out, sed_label, doa_out, doa_label, per_class=False):
    sed_label = sed_label.to(sed_out.device)
    doa_label = doa_label.to(doa_out.device)
    _n_class = sed_out.shape[-1]
    # Calculate LE
    mask = torch.norm(doa_label, dim=-1) > 0.5   # Active sound event mask                   # [B, S, T]
    cos_sim = torch.sum(F.normalize(doa_out, dim=-1) * F.normalize(doa_label, dim=-1), dim=-1)      # [B, S, T]
    angles = torch.rad2deg(torch.acos(torch.clamp(cos_sim, -1.0, 1.0)))                                            # [B, S, T]
    # Calculate TP, TN, FP
    TP = (sed_out * sed_label)   # True Positives                                        # [B, S, T, C]
    FP = (sed_out * (1 - sed_label))     # False Positives                               # [B, S, T, C]
    FN = ((1 - sed_out) * sed_label)     # False Negatives                               # [B, S, T, C]
    # Spatial Threshold
    within_threshold = (angles <= 20).unsqueeze(3).float().to(sed_label.device) * sed_label  # Localization Recall within threshold         # [B, S, T, C]
    TP_wi_th = TP * within_threshold    # True Positives within threshold   # [B, S, T, C]
    FP_wi_th = FP + (TP - TP_wi_th)     # False Positives within threshold  # [B, S, T, C]
    # Class-wise TP, TN, FP
    TP_c = TP.sum(dim=[0, 1, 2])
    TP_c_wi_th = TP_wi_th.sum(dim=[0, 1, 2])
    FP_c = FP.sum(dim=[0, 1, 2])
    FN_c = FN.sum(dim=[0, 1, 2])
    angles_c = (angles.unsqueeze(-1) * sed_label).sum(dim=[0, 1, 2])
    mask_c = (mask.unsqueeze(-1) * sed_label).sum(dim=[0, 1, 2])
    # Calculate ER, F1
    ER = (torch.minimum(FP_wi_th, FN).sum() + torch.clamp(FP_wi_th - FN, min=0).sum() + torch.clamp(FN - FP_wi_th, min=0).sum()) / (TP.sum() + FN.sum() + 1e-8)
    precision = TP_c_wi_th / (TP_c_wi_th + FP_wi_th.sum(dim=[0, 1, 2]) + 1e-8)
    recall = TP_c_wi_th / (TP_c + FN_c + 1e-8)
    F1_c = 2 * precision * recall / (precision + recall + 1e-8)
    LE_c = angles_c / (mask_c + 1e-8)
    LR_c = TP_c / (TP_c + FN_c + 1e-8)
    # macro average
    F1 = F1_c.mean()
    LE = LE_c.mean()
    LR = LR_c.mean()
    seld_score = (ER + (1 - F1) + LE / 180 + (1 - LR)) / 4
    if per_class:
        return 100*ER.item(), 100*F1.item(), LE.item(), 100*LR.item(), seld_score.item(), F1_c, LE_c, LR_c
    else:
        return 100*ER.item(), 100*F1.item(), LE.item(), 100*LR.item(), seld_score.item()

def validate():
    return SELD_metrics