import torch
import torch.nn as nn
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell


class L2MU(nn.Module):
    """
    Experiment 2D model.

    Runs the frozen backbone (m + y populations) across all T timesteps
    inside torch.no_grad(), accumulates spike rates, then applies the
    trainable W_out once — outside no_grad — so CE loss gradients reach
    only W_out.weight and nothing else.

    This is the correct way to train only a linear readout on top of a
    frozen spiking backbone without needing surrogate gradients at all:
    the gradient of CE w.r.t. W_out.weight is simply spk_y_rate itself.

    Forward returns:
        logits      [B, 27]       — fed to cross_entropy loss
        spk_y_stack [T, B, 250]   — for monitoring y rates during training
    """

    def __init__(self, input_size, params, neuron_type='Leaky'):
        super().__init__()
        self.l2mu_cell = L2MUCell(
            input_size=input_size,
            params=params,
            neuron_type=neuron_type,
        )
        self.num_classes = L2MUCell.NUM_CLASSES

    def forward(self, input_seq: torch.Tensor):
        """
        Args:
            input_seq : [T, B, input_size]
        Returns:
            logits      : [B, 27]
            spk_y_stack : [T, B, memory_size]
        """
        T = input_seq.size(0)
        spk_memory = self.l2mu_cell.init_cell()
        spk_y_list = []

        # ── Backbone: run all T steps, no gradients needed ────────────────
        # D and W_in are frozen. We also block grads through the LIF states
        # so backprop is clean and fast (no BPTT through 256 timesteps).
        with torch.no_grad():
            for step in range(T):
                spk_y, spk_memory = self.l2mu_cell(
                    input_seq[step].flatten(1), spk_memory)
                spk_y_list.append(spk_y)

        spk_y_stack = torch.stack(spk_y_list, dim=0)   # [T, B, 250]

        # Spike rate: normalise by T so values are in [0, 1] regardless
        # of sequence length — keeps W_out inputs well-scaled for Adam.
        spk_y_rate  = spk_y_stack.mean(dim=0)           # [B, 250]

        # ── Classifier: W_out applied OUTSIDE no_grad ─────────────────────
        # spk_y_rate has requires_grad=False (computed inside no_grad).
        # logits still has requires_grad=True because W_out.weight does.
        # d(loss)/d(W_out.weight) = spk_y_rate.T @ d(loss)/d(logits)
        logits = self.l2mu_cell.W_out(spk_y_rate)       # [B, 27]

        return logits, spk_y_stack

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Argmax of logits → predicted class index."""
        return logits.argmax(dim=1)