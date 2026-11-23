"""
Warmup + ReduceLROnPlateau scheduler.
Linear warmup for the first N epochs, then delegates to ReduceLROnPlateau.
Subclasses torch LRScheduler so Lightning accepts it without lr_scheduler_step override.
"""

from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class WarmupReduceLROnPlateau(LRScheduler):
    """
    First applies linear warmup from (max_lr / warmup_epochs) to max_lr over
    warmup_epochs, then switches to ReduceLROnPlateau.

    Despite the name, warmup_epochs counts calls to .step(), not calendar
    epochs: it's the number of Lightning `lr_scheduler.interval` ticks to warm
    up over. Set interval="epoch" for epoch-based warmup, or interval="step"
    (with warmup_epochs equal to the desired step count) for step-based warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of scheduler.step() calls to warm up over (LR grows linearly to max_lr).
        max_lr: Peak learning rate after warmup (and initial base for plateau).
        mode: 'min' or 'max' for ReduceLROnPlateau.
        factor: LR reduction factor for ReduceLROnPlateau.
        patience: Patience (epochs) for ReduceLROnPlateau.
        min_lr: Minimum LR for ReduceLROnPlateau.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_lr: float,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self._warmup_step_count = 0
        self._last_plateau_metric = None

        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Return current LR (from optimizer). Used by base for _last_lr when needed."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metric: Optional[float] = None, epoch: Optional[int] = None):
        """Step with optional metric (for ReduceLROnPlateau phase)."""
        self._warmup_step_count += 1
        if self._warmup_step_count <= self.warmup_epochs:
            lr = self.max_lr * self._warmup_step_count / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # ReduceLROnPlateau.step(metric) requires a float; if metric is None (e.g. val not run),
            # skip plateau step so training continues without crashing.
            # Also skip if the metric hasn't changed since the last call: with step-based
            # warmup (interval="step"), Lightning invokes step() every training step and
            # keeps re-supplying the last logged epoch metric until the next validation
            # epoch, which would otherwise make patience count in steps instead of epochs.
            if metric is not None and metric != self._last_plateau_metric:
                self.plateau.step(metric)
                self._last_plateau_metric = metric
            # else: keep current LR and continue training
        self.last_epoch = self._warmup_step_count - 1
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """For checkpoint save/restore."""
        state = super().state_dict()
        state["_warmup_step_count"] = self._warmup_step_count
        state["warmup_epochs"] = self.warmup_epochs
        state["max_lr"] = self.max_lr
        state["_last_plateau_metric"] = self._last_plateau_metric
        state["plateau_state_dict"] = self.plateau.state_dict()
        return state

    def load_state_dict(self, state_dict: dict):
        """For checkpoint save/restore."""
        plateau_state = state_dict.pop("plateau_state_dict", None)
        self._warmup_step_count = state_dict.pop("_warmup_step_count", 0)
        self.warmup_epochs = state_dict.pop("warmup_epochs", self.warmup_epochs)
        self.max_lr = state_dict.pop("max_lr", self.max_lr)
        self._last_plateau_metric = state_dict.pop("_last_plateau_metric", None)
        super().load_state_dict(state_dict)
        if plateau_state is not None:
            self.plateau.load_state_dict(plateau_state)
