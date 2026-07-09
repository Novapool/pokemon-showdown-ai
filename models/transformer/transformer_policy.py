"""
transformer_policy.py — Shared transformer policy/value network for the
(12, 65) structured observation.

Architecture (M3 spec in MILESTONES.md):
    Linear(65 -> d_model=128)
    TransformerEncoder(layers=2, nhead=4, d_ff=256, dropout=0.1)
    mean-pool over non-unknown tokens -> (batch, 128)
    policy head: Linear(128, 9), value head: Linear(128, 1)

Unknown opponent tokens (unknown_flag == 1) are attention-masked via
key_padding_mask and excluded from the pooled context, per the M3 spec.

This module is shared between BC pretraining (models/bc_pretrain.py) and the
M3 PPO training loop, so BC checkpoints load into PPO without key remapping.
`load_pretrain_checkpoint()` implements the tolerant load: any tensor whose
name or shape doesn't match the target model is skipped with a warning.
"""

import torch
import torch.nn as nn

N_TOKENS = 12
TOKEN_DIM = 65
N_ACTIONS = 9
UNKNOWN_FLAG = 39  # token feature index, see models/metamon_adapter.py layout


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        n_actions: int = N_ACTIONS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1)

        self.hparams = dict(
            token_dim=token_dim,
            n_actions=n_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: float32 (batch, 12, 65) -> (logits (batch, 9), value (batch,))."""
        pad_mask = obs[:, :, UNKNOWN_FLAG] > 0.5  # True = mask out
        x = self.embed(obs)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        context = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.policy_head(context), self.value_head(context).squeeze(-1)


def load_pretrain_checkpoint(model: nn.Module, path: str) -> int:
    """Load BC-pretrained weights into `model`, skipping mismatched layers.

    Any checkpoint tensor whose name is absent from the model, or whose shape
    differs, is skipped with a warning instead of raising. Returns the number
    of tensors actually loaded.
    """
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()

    loadable = {}
    for name, tensor in source.items():
        if name not in target:
            print(f"[pretrain] WARNING: skipping '{name}' — not in model")
        elif target[name].shape != tensor.shape:
            print(
                f"[pretrain] WARNING: skipping '{name}' — shape mismatch "
                f"(checkpoint {tuple(tensor.shape)} vs model {tuple(target[name].shape)})"
            )
        else:
            loadable[name] = tensor

    missing = [name for name in target if name not in loadable]
    if missing:
        print(f"[pretrain] WARNING: {len(missing)} model tensors not in checkpoint: {missing}")

    model.load_state_dict(loadable, strict=False)
    print(f"[pretrain] loaded {len(loadable)}/{len(target)} tensors from {path}")
    return len(loadable)
