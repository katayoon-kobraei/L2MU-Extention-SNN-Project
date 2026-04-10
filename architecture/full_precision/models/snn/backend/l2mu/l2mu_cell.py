import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Parameters removed vs original L2MU:
        - hidden_size, e_x, e_h, e_m, W_x, W_h, W_m, spk_u, spk_h
    Parameters kept/added:
        - A, B  (fixed, LMU)
        - C     (fixed, LDN Legendre projection)
        - D     (trainable, generic weight matrix)
        - W_out (trainable)
        - spk_m, spk_y, spk_out  (three LIF populations)
    """

    def __init__(
            self,
            input_size,
            output_size,
            params,
            trainable_theta=False,
            neuron_type='Leaky',
    ):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=output_size,
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )

        self.init_parameters()   # builds A, B, C (fixed) + D, W_out (trainable)

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(
                f"Neuron type '{neuron_type}' not found in snntorch."
            )

        # m population — one LIF neuron per (memory_size × order) state element
        self.spk_m = Neuron(
            beta=params['beta_spk_m'],
            threshold=params['threshold_spk_m'],
            learn_beta=True,
            learn_threshold=True,
            init_hidden=True,
        )

        # y population — one LIF neuron per memory slot
        self.spk_y = Neuron(
            beta=params['beta_spk_y'],
            threshold=params['threshold_spk_y'],
            learn_beta=True,
            learn_threshold=True,
            init_hidden=True,
        )

        # out population — one LIF neuron per output class
        self.spk_out = Neuron(
            beta=params['beta_spk_out'],
            threshold=params['threshold_spk_out'],
            learn_beta=True,
            learn_threshold=True,
            init_hidden=True,
        )

    # ------------------------------------------------------------------
    def init_cell(self):
        """Reset all LIF membrane potentials and return empty state."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        # spk_memory starts as empty — zeros are created on first forward
        return torch.empty(0)   # spk_memory placeholder

    # ------------------------------------------------------------------
    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        One time-step forward pass.

        Args:
            spk_input  : binary spikes from the input layer [B, input_size]
            spk_memory : previous memory spikes  [B, memory_size, order]
                         (empty tensor on the very first step)

        Returns:
            spk_out    : output spikes for loss computation [B, output_size]
            spk_memory : updated memory spikes [B, memory_size, order]
        """
        batch_size = spk_input.shape[0]

        # Initialise memory state to zeros on first call
        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (batch_size, self.memory_size, self.order),
                dtype=torch.float,
                device=spk_input.device,
            )

        # ── m population ──────────────────────────────────────────────
        # spk_input: [B, input_size] → unsqueeze → [B, input_size, 1]
        # B(spk_input):  [B, memory_size, 1]   (broadcast over memory slots)
        # A(spk_memory): [B, memory_size, order] → stays same shape
        spk_input_3d = spk_input.unsqueeze(-1)          # [B, input_size, 1]

        curr_m = self.A(spk_memory) + self.B(spk_input_3d)  # [B, memory_size, order]

        if self.discretizer == 'euler' and self.trainable_theta:
            curr_m = curr_m + curr_m * self.theta_inv

        spk_memory = self.spk_m(curr_m)                 # [B, memory_size, order]

        # ── y population ──────────────────────────────────────────────
        # C(spk_memory): [B, memory_size, 1]  — Legendre projection per slot
        # D(spk_input):  [B, memory_size]     — trainable direct input path
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        #           [B, memory_size]                 [B, memory_size]

        spk_y = self.spk_y(curr_y)                      # [B, memory_size]

        # ── out population ────────────────────────────────────────────
        curr_out = self.W_out(spk_y)                    # [B, output_size]
        spk_out  = self.spk_out(curr_out)               # [B, output_size]

        return spk_out, spk_memory
