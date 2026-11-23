"""Evaluate a trained model on the ASA test split.

Per-sample metrics follow the validation pipeline used during training:
  1) PIT-based output permutation (from the ASA loss function),
  2) SED map built from on/off * argmax(class) for each object,
  3) SELD metrics (ER / F1 / LE / LR / SELD score) and SI-SDRi.
Reported numbers are the mean over test samples.
"""
import os
import sys
import json
import pathlib
import argparse

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import LABELS, parse_yaml, initialize_config, ignore_warnings
ignore_warnings()

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr

ALL_LABELS = LABELS['ASA2'] + ['silence']   # index -1 -> 'silence'
METRIC_KEYS = ['si_sdri', 'ane_si_sdri', 'ER', 'F1', 'LE', 'LR', 'SELD_score']


def build_from_train_config(config_yaml, data_dir, split):
    """Re-use the model / dataset definitions of a training config for evaluation."""
    cfg = parse_yaml(config_yaml)
    model_cfg = cfg['lightning_module']['args']['model']
    model_cfg['args'].pop('ckpt', None)  # backbone init is only needed for training

    ds_cfg = cfg['datamodule']['args']['val_dataloader']['dataset']
    ds_cfg['args']['dataset'] = split
    if data_dir:
        ds_cfg['args']['asa_dir'] = data_dir

    model = initialize_config(model_cfg)
    dataset = initialize_config(ds_cfg)
    loss_func = initialize_config({'module': 'src.training.loss.asa.asa', 'main': 'get_loss_func'})
    metric_func = initialize_config(cfg['lightning_module']['args']['metric'])
    return model, dataset, loss_func, metric_func


def load_weights(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(state_dict, strict=True)
    return model


def si_sdri(est, tar, mix):
    """est, tar: [S, M, L], mix: [M, L] -> mean SI-SDRi over sources and channels."""
    mix = mix.unsqueeze(0).expand_as(tar)
    return (si_snr(est, tar) - si_snr(mix, tar)).mean()


@torch.no_grad()
def evaluate(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model, dataset, loss_func, metric_func = build_from_train_config(args.config, args.data_dir, args.split)
    model = load_weights(model, args.ckpt).to(device).eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=dataset.collate_fn, num_workers=args.num_workers)

    out_dir = args.output_dir or os.path.join(
        'results', f'{pathlib.Path(args.config).stem}_{pathlib.Path(args.ckpt).stem}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for batch in tqdm(dataloader):
        index = str(batch['paras'][0]['index'])
        n_src = int(batch['paras'][0]['n_src'])

        output = model({'mixture': batch['mixture'].to(device)})
        target = {k: batch[k].to(device) for k in ('anechoic', 'reverb', 'noise', 'cls_label', 'sed_label', 'doa_label')}
        target['onoff_label'] = (torch.norm(target['doa_label'], dim=-1) > 0.5).float()

        _, perm = loss_func(output, target)

        # SED map: frame activity (on/off) x argmax class, per object
        active = ((torch.norm(perm['doa_label'], dim=-1) > 0.5) * (torch.sigmoid(perm['onoff_label']) > 0.5)).float()
        cls_onehot = torch.zeros_like(perm['cls_label']).scatter_(-1, perm['cls_label'].argmax(-1, keepdim=True), 1)
        perm['sed_label'] = active.unsqueeze(-1) * cls_onehot.unsqueeze(2)

        m = metric_func(perm, target)

        # separation metrics over the active (non-padded) objects
        mix = batch['mixture'][0].to(device)                                    # [M, L]
        tar_ane = target['anechoic'][0, :n_src]
        tar_sep = tar_ane + target['reverb'][0, :n_src]                         # [S, M, L]
        est_ane = perm['anechoic'][0, :n_src]
        est_sep = est_ane + perm['reverb'][0, :n_src]

        results[index] = {
            'n_src': n_src,
            'si_sdri': si_sdri(est_sep, tar_sep, mix).item(),
            'ane_si_sdri': si_sdri(est_ane, tar_ane, mix).item(),
            'ER': m['ER'].item(),
            'F1': m['F1'].item(),
            'LE': m['LE'].item(),
            'LR': m['LR'].item(),
            'SELD_score': m['seld_score'].item(),
        }

        if args.save_audio:
            sample_dir = os.path.join(out_dir, 'samples', index)
            os.makedirs(sample_dir, exist_ok=True)
            est_cls = perm['cls_label'][0].argmax(-1)
            est_cls[active[0].max(-1).values == 0] = -1
            sf.write(os.path.join(sample_dir, 'mixture.wav'), mix.T.cpu().numpy(), dataset.sr)
            for s in range(est_sep.shape[0]):
                name = ALL_LABELS[int(est_cls[s])]
                sf.write(os.path.join(sample_dir, f'{s}_est_{name}.wav'), est_sep[s].T.cpu().numpy(), dataset.sr)
                sf.write(os.path.join(sample_dir, f'{s}_tar_{ALL_LABELS[int(target["cls_label"][0, s])]}.wav'),
                         tar_sep[s].T.cpu().numpy(), dataset.sr)

    summary = {k: float(np.mean([r[k] for r in results.values()])) for k in METRIC_KEYS}
    for n in sorted({r['n_src'] for r in results.values()}):
        rs = [r for r in results.values() if r['n_src'] == n]
        summary[f'n_src={n}'] = {k: float(np.mean([r[k] for r in rs])) for k in METRIC_KEYS} | {'count': len(rs)}

    with open(os.path.join(out_dir, 'metrics_per_sample.json'), 'w') as f:
        json.dump(dict(sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])), f, indent=4)
    with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f'\n=== {args.ckpt} ({args.split}, {len(results)} samples) ===')
    for k in METRIC_KEYS:
        print(f'{k:>12s}: {summary[k]:.4f}')
    print(f'Saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='Training config of the model.')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint (.ckpt) to evaluate.')
    parser.add_argument('--data_dir', type=str, default='', help='Override `asa_dir` in the config.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split directory name.')
    parser.add_argument('--output_dir', '-o', type=str, default='', help='Where to save metrics (default: results/...).')
    parser.add_argument('--save_audio', action='store_true', help='Also write estimated / reference wavs.')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    evaluate(parser.parse_args())
