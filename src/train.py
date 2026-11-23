import os
import sys
import json
import pathlib
import argparse
import logging

# Make `src.*` importable regardless of the current working directory.
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import logging_setup, parse_yaml, initialize_config, ignore_warnings
ignore_warnings()

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


def train(args):
    configs = parse_yaml(args.config_yaml)
    workspace = args.workspace
    log_name = args.log_name or pathlib.Path(args.config_yaml).stem

    # deterministic
    if configs.get('deterministic', False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        pl.seed_everything(configs['manual_seed'], workers=True)
        configs['train']['trainer']['args']['deterministic'] = True

    # workspace directories
    checkpoints_dir, logs_dir, tf_logs_dir, wandb_dir = (
        os.path.join(workspace, d) for d in ('checkpoints', 'logs', 'tf_logs', 'wandb'))
    for d in (checkpoints_dir, logs_dir, tf_logs_dir):
        os.makedirs(d, exist_ok=True)

    if rank_zero_only.rank == 0:
        logging_setup(directory=logs_dir, console_log_level=args.log_level.upper())
        logger.info(args)
        logger.info(json.dumps(configs, indent=4))

    # optional overrides
    dl_cfg = configs['datamodule']['args']
    if args.batch_size > 0:
        for split in ('train_dataloader', 'val_dataloader'):
            if split in dl_cfg:
                dl_cfg[split]['batch_size'] = args.batch_size
    if args.data_dir:
        for split in ('train_dataloader', 'val_dataloader'):
            if split in dl_cfg:
                dl_cfg[split]['dataset']['args']['asa_dir'] = args.data_dir
    if args.devices:
        configs['train']['trainer']['args']['devices'] = [int(d) for d in args.devices.split(',')]

    data_module = initialize_config(configs['datamodule'])
    pl_model = initialize_config(configs['lightning_module'])

    # callbacks
    callbacks = []
    for callback_config in configs['train']['callbacks']:
        if callback_config['name'] == 'checkpoint':
            callback_config['args']['dirpath'] = checkpoints_dir
        callbacks.append(initialize_config(callback_config))

    # loggers
    loggers = [pl.loggers.TensorBoardLogger(save_dir=tf_logs_dir, name=log_name, version=args.version)]
    if args.wandb_project:
        os.makedirs(wandb_dir, exist_ok=True)
        loggers.append(pl.loggers.WandbLogger(
            entity=args.wandb_entity, project=args.wandb_project, name=log_name,
            save_dir=wandb_dir, log_model=False))

    configs['train']['trainer']['args']['callbacks'] = callbacks
    configs['train']['trainer']['args']['logger'] = loggers
    trainer = initialize_config(configs['train']['trainer'])

    # resume
    if args.resume_last:
        ckpt_path = os.path.join(checkpoints_dir, 'last.ckpt')
    elif args.resume_checkpoint_path:
        ckpt_path = args.resume_checkpoint_path
    else:
        ckpt_path = None
    if ckpt_path:
        logger.info(f'Resume training from: {ckpt_path}')

    trainer.fit(model=pl_model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", "-c", type=str, required=True, help="Path of the training config.")
    parser.add_argument("--workspace", "-w", type=str, default='workspace/test', help="Output directory (checkpoints, logs).")
    parser.add_argument("--data_dir", type=str, default='', help="Override `asa_dir` of the train/val datasets.")
    parser.add_argument("--devices", type=str, default='', help="Comma-separated GPU ids, e.g. '0,1,2,3'.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Override per-GPU batch size of train/val.")
    parser.add_argument("--version", "-v", type=str, default='0', help="TensorBoard logger version.")
    parser.add_argument("--log_name", type=str, default='', help="Logger run name (default: config file name).")
    parser.add_argument("--resume_checkpoint_path", "-r", type=str, default='', help="Resume from this checkpoint.")
    parser.add_argument("--resume_last", action='store_true', help="Resume from <workspace>/checkpoints/last.ckpt.")
    parser.add_argument("--wandb_project", type=str, default='', help="Enable W&B logging with this project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (optional).")
    parser.add_argument("--log_level", "-l", type=str, default='INFO')
    train(parser.parse_args())
