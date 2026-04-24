"""
Flat decoder MLP surrogate for the cardiovascular ODE simulator.

Takes a 25-dim parameter vector and produces:
  - 24 continuous waveforms of length 201 (z-score normalised space)
  - 4 binary valve signals of length 201 (raw logits — apply sigmoid to read)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CVSurrogate(nn.Module):
    """
    Architecture
    ------------
    params (B, 25) → shared trunk (Linear → SiLU × n_layers, width=hidden)
                   → cont  head: Linear(hidden, n_cont  × T) → (B, n_cont,  T)
                   → valve head: Linear(hidden, n_valve × T) → (B, n_valve, T)

    The valve head emits raw logits (loss uses BCEWithLogitsLoss); apply a
    sigmoid at inference time to read them as probabilities.
    """

    def __init__(
        self,
        n_params: int = 25,
        n_cont: int = 24,
        n_valve: int = 4,
        T: int = 201,
        hidden: int = 1024,
        n_layers: int = 6,
    ):
        super().__init__()
        self.n_params = n_params
        self.n_cont = n_cont
        self.n_valve = n_valve
        self.T = T

        # ── Shared trunk ─────────────────────────────────────────────────
        trunk_layers: list[nn.Module] = [
            nn.Linear(n_params, hidden),
            nn.SiLU(),
        ]
        for _ in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden, hidden))
            trunk_layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # ── Output heads ─────────────────────────────────────────────────
        self.cont_head = nn.Linear(hidden, n_cont * T)
        self.valve_head = nn.Linear(hidden, n_valve * T)

    def forward(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(params)                           # (B, hidden)

        cont = self.cont_head(h).view(-1, self.n_cont, self.T)
        valve = self.valve_head(h).view(-1, self.n_valve, self.T)

        return cont, valve


class CVLoss(nn.Module):
    """
    MSE on continuous waveforms + BCE on valve logits.

    `forward` returns (total, cont_loss, valve_loss) so each component can be
    logged separately.
    """

    def __init__(self, valve_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.valve_weight = valve_weight

    def forward(
        self,
        pred_cont: torch.Tensor,
        pred_valve: torch.Tensor,
        target_cont: torch.Tensor,
        target_valve: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_cont = self.mse(pred_cont, target_cont)
        loss_valve = self.bce(pred_valve, target_valve)
        total = loss_cont + self.valve_weight * loss_valve
        return total, loss_cont, loss_valve