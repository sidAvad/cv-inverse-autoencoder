"""
CVEncoder: inverse MLP surrogate for the cardiovascular ODE simulator.

Takes flattened waveforms (continuous z-scored + binarized valve) and
predicts the 25-dim z-scored parameter vector.

Architecture mirrors the decoder trunk depth:
  waveforms (5628,) → trunk (6 × Linear(1024) + SiLU) → Linear(1024, 25)
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_CONT  = 24
N_VALVE = 4
T       = 201
N_INPUT = (N_CONT + N_VALVE) * T   # 5628
N_PARAMS = 25


class CVEncoder(nn.Module):
    """
    Architecture
    ------------
    waveforms (B, 5628) → trunk (Linear → SiLU × n_layers, width=hidden)
                        → param head: Linear(hidden, 25)

    Inputs:
      - Continuous waveforms: z-scored, shape (B, 24, 201)
      - Valve signals: binarized {0,1}, shape (B, 4, 201)
    Both are flattened and concatenated before the trunk.
    """

    def __init__(
        self,
        n_input:  int = N_INPUT,
        n_params: int = N_PARAMS,
        hidden:   int = 1024,
        n_layers: int = 6,
    ):
        super().__init__()

        trunk_layers: list[nn.Module] = [
            nn.Linear(n_input, hidden),
            nn.SiLU(),
        ]
        for _ in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden, hidden))
            trunk_layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*trunk_layers)

        self.param_head = nn.Linear(hidden, n_params)

    def forward(
        self,
        waves_cont:  torch.Tensor,  # (B, 24, 201) — z-scored
        waves_valve: torch.Tensor,  # (B, 4, 201)  — {0, 1}
    ) -> torch.Tensor:              # (B, 25)       — z-scored params
        valve_bin = (waves_valve > 0.5).float()
        x = torch.cat([
            waves_cont.flatten(1),   # (B, 4824)
            valve_bin.flatten(1),    # (B, 804)
        ], dim=1)                    # (B, 5628)
        return self.param_head(self.trunk(x))
