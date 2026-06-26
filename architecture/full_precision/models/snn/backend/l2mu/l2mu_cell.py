import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore
from architecture.full_precision.core.lmu.utils import XavierLinear


class L2MUCell(LMUCore):
    """
    Experiment 2D: Frozen backbone + W_out trained by backpropagation.

    Identical backbone to Exp2C (W_in and D frozen from Exp1).
    No WTA. No STDP. W_out is a standard trainable linear layer.

    The model (l2mu_exp2d.py) runs the backbone inside torch.no_grad(),
    accumulates spike rates over T, then applies W_out outside no_grad
    so cross-entropy loss gradients flow only to W_out.weight.

    Architecture:
        m population : m[t+1] = A*m[t] + B*W_in(x[t])   (W_in frozen)
        y population : y[t]   = C*m[t] + D*x[t]          (D    frozen)
        classifier   : logits  = W_out * mean_t(spk_y)    (W_out ← Adam + CE)

    Changes vs l2mu_cell_exp2c.py
    ──────────────────────────────
    removed : WTA block
    removed : S2-STDP constants, s2stdp_update(), neuron_roles/neuron_class buffers
    removed : spk_out / spk_out population entirely
    changed : output_size = 27  (was 54)
    changed : W_out.requires_grad = True  (was False — STDP updated it manually)
    forward : returns (spk_y, spk_memory) only — model handles W_out separately
    """

    NUM_CLASSES = 27

    def __init__(self, input_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=self.NUM_CLASSES,       # 27
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()
        self.e_x.weight.requires_grad_(False) 

        # ── D [24 → 250]: override base interface, freeze, load from Exp1 ──
        self.D = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.D.weight, 0.0, 1.0)
        self.D.weight.requires_grad_(False)

        # ── W_in [24 → 250]: same, freeze, load from Exp1 ────────────────
        self.W_in = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.W_in.weight, 0.0, 1.0)
        self.W_in.weight.requires_grad_(False)

        # ── W_out [250 → 27]: TRAINABLE — updated by optimizer, not STDP ─
        torch.nn.init.xavier_normal_(self.W_out.weight)
        self.W_out.weight.requires_grad_(True)   # ← key difference from STDP exps

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"Neuron type '{neuron_type}' not found in snntorch.")

        self.spk_m = Neuron(
            beta=params['beta_spk_m'], threshold=params['threshold_spk_m'],
            learn_beta=False, learn_threshold=False, init_hidden=True,
        )
        self.spk_y = Neuron(
            beta=params['beta_spk_y'], threshold=params['threshold_spk_y'],
            learn_beta=False, learn_threshold=False, init_hidden=True,
        )
        # No spk_out neuron — W_out outputs logits directly, not spikes

    # ── Weight loading ─────────────────────────────────────────────────────

    def load_D_from_exp1(self, D_weights_path):
        D_weights = torch.load(D_weights_path, map_location='cpu')
        self.D.weight.data.copy_(D_weights)
        self.D.weight.requires_grad_(False)
        print(f"Loaded D     : shape={tuple(self.D.weight.shape)}  "
              f"mean={self.D.weight.mean():.4f}  std={self.D.weight.std():.4f}")

    def load_W_in_from_exp1(self, W_in_weights_path):
        W_in_weights = torch.load(W_in_weights_path, map_location='cpu')
        self.W_in.weight.data.copy_(W_in_weights)
        self.W_in.weight.requires_grad_(False)
        print(f"Loaded W_in  : shape={tuple(self.W_in.weight.shape)}  "
              f"mean={self.W_in.weight.mean():.4f}  std={self.W_in.weight.std():.4f}")

    # ── Core methods ───────────────────────────────────────────────────────

    def init_cell(self):
        """Reset LIF states. Call once per sequence."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        return torch.empty(0)

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        One-timestep backbone forward pass.
        W_out is NOT applied here — the model (l2mu_exp2d.py) applies it
        after accumulating spike rates over all T steps.

        Args:
            spk_input  : [B, input_size]
            spk_memory : [B, memory_size, order]
        Returns:
            spk_y      : [B, memory_size]
            spk_memory : [B, memory_size, order]
        """
        B, device = spk_input.shape[0], spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (B, self.memory_size, self.order), dtype=torch.float, device=device)

        # m population
        u_t        = self.W_in(spk_input)
        curr_m     = self.A(spk_memory) + self.B(u_t.unsqueeze(-1))
        spk_memory = self.spk_m(curr_m)

        # y population
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        spk_y  = self.spk_y(curr_y)               # [B, 250]

        return spk_y, spk_memory