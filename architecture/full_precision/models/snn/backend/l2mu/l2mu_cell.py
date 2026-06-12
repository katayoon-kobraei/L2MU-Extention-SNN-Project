import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Experiment 2A: SSM core with m, y, and out populations.
    D is frozen (loaded from Exp1).
    W_out (27 neurons, one per class) trained with unsupervised STDP.
    Label assignment done after training.

    Architecture:
        m population: m[t+1] = A * m[t] + B * e_x(x[t])  (fixed)
        y population: y[t]   = C * m[t] + D * x[t]        (D frozen from Exp1)
        out population: o[t] = W_out * spk_y[t]            (W_out STDP)
    """

    # STDP for W_out
    STDP_A_PLUS   = 0.02
    STDP_A_MINUS  = 0.02
    STDP_TAU_PRE  = 0.95
    STDP_TAU_POST = 0.95

    W_MAX = 1.0
    W_MIN = 0.0

    NUM_CLASSES = 27  # Braille letters

    def __init__(self, input_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=self.NUM_CLASSES,   # 27 output neurons
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()

        # e_x: fixed random projection (same as Exp1)
        self.W_in.weight.requires_grad_(False)

        # D: frozen — loaded from Exp1
        self.D.weight.requires_grad_(False)

        # W_out: trained by STDP — excluded from optimizer
        self.W_out.weight.requires_grad_(False)

        # Initialize W_out weights uniformly in [0, 1]
        torch.nn.init.uniform_(self.W_out.weight, 0.0, 1.0)

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"Neuron type '{neuron_type}' not found in snntorch.")

        # m population
        self.spk_m = Neuron(
            beta=params['beta_spk_m'],
            threshold=params['threshold_spk_m'],
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # y population
        self.spk_y = Neuron(
            beta=params['beta_spk_y'],
            threshold=params['threshold_spk_y'],
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # out population — 27 LIF neurons, one per class
        self.spk_out = Neuron(
            beta=params['beta_spk_out'],
            threshold=params['threshold_spk_out'],
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # STDP traces for W_out (pre=spk_y, post=spk_out)
        self.trace_pre_Wout  = None
        self.trace_post_Wout = None

        # Spike history
        self._spk_y_hist   = []
        self._spk_out_hist = []

    def load_D_from_exp1(self, D_weights_path):
        """Load D weights from Experiment 1 and freeze."""
        D_weights = torch.load(D_weights_path, map_location='cpu')
        self.D.weight.data.copy_(D_weights)
        self.D.weight.requires_grad_(False)
        print(f"Loaded D weights from {D_weights_path}")
        print(f"D weight mean: {self.D.weight.mean().item():.4f}")
    
    def load_W_in_from_exp1(self, W_in_weights_path):
        """Load W_in weights from Experiment 1 and freeze."""
        W_in_weights = torch.load(W_in_weights_path, map_location='cpu')
        self.W_in.weight.data.copy_(W_in_weights)
        self.W_in.weight.requires_grad_(False)
        print(f"Loaded W_in weights from {W_in_weights_path}")
        print(f"W_in weight mean: {self.W_in.weight.mean().item():.4f}")

    def init_cell(self):
        """Reset LIF states and STDP traces — called once per batch."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        self.trace_pre_Wout  = None
        self.trace_post_Wout = None
        self._spk_y_hist   = []
        self._spk_out_hist = []
        return torch.empty(0)

    def _init_traces(self, batch_size, device):
        self.trace_pre_Wout  = torch.zeros(batch_size, self.memory_size, device=device)
        self.trace_post_Wout = torch.zeros(batch_size, self.NUM_CLASSES, device=device)

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        batch_size = spk_input.shape[0]
        device     = spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (batch_size, self.memory_size, self.order),
                dtype=torch.float, device=device)

        if self.trace_pre_Wout is None:
            self._init_traces(batch_size, device)

        # m population
        u_t    = self.W_in(spk_input)            # W_in replaces e_x
        u_t_3d = u_t.unsqueeze(-1)
        curr_m = self.A(spk_memory) + self.B(u_t_3d)
        spk_memory = self.spk_m(curr_m)

        # y population — D frozen from Exp1
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)  # D gets raw input
        spk_y  = self.spk_y(curr_y)             # [B, memory_size]

        # out population — W_out trained by STDP
        curr_out = self.W_out(spk_y)            # [B, 27]
        spk_out  = self.spk_out(curr_out)       # [B, 27]

        # Save for STDP
        self._spk_y_hist.append(spk_y.detach())
        self._spk_out_hist.append(spk_out.detach())

        return spk_out, spk_y, spk_memory

    def stdp_update(self):
        """
        Weight-dependent STDP for W_out only.
        pre = spk_y (y population)
        post = spk_out (output population)
        """
        if not self._spk_y_hist:
            return

        with torch.no_grad():
            w = self.W_out.weight   # [27, memory_size]

            for spk_y, spk_out in zip(
                self._spk_y_hist,
                self._spk_out_hist,
            ):
                # Update traces
                self.trace_pre_Wout  = (
                    self.STDP_TAU_PRE  * self.trace_pre_Wout  + spk_y
                )   # [B, memory_size]
                self.trace_post_Wout = (
                    self.STDP_TAU_POST * self.trace_post_Wout + spk_out
                )   # [B, 27]

                # Potentiation: out fires → strengthen from active y
                dW_plus = self.STDP_A_PLUS * torch.einsum(
                    'bi,bj->ij', spk_out, self.trace_pre_Wout
                ) / spk_y.shape[0]   # [27, memory_size]

                # Depression: y fires → weaken toward active out
                dW_minus = self.STDP_A_MINUS * torch.einsum(
                    'bi,bj->ij', self.trace_post_Wout, spk_y
                ) / spk_y.shape[0]  # [27, memory_size]

                # Weight-dependent update
                dW = dW_plus * (self.W_MAX - w) - dW_minus * (w - self.W_MIN)
                self.W_out.weight.add_(dW)

            # Clip weights
            self.W_out.weight.clamp_(self.W_MIN, self.W_MAX)

        # Clear history
        self._spk_y_hist   = []
        self._spk_out_hist = []