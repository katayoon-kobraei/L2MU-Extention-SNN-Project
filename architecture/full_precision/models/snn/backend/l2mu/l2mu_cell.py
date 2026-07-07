import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore
from architecture.full_precision.core.lmu.utils import XavierLinear


class L2MUCell(LMUCore):
    """
    Experiment 1 — Simultaneous W_in + D STDP with WTA (corrected run).

    Prof. Fra's instruction: train W_in with the same STDP strategy as D.
    The previous simultaneous run failed because WTA was missing — all y
    neurons converged to the same average response without competition.
    WTA is the fix: only the winning y neuron's weights update per step,
    forcing each neuron to specialize toward a different Braille pattern.

    Architecture:
        m population : m[t+1] = A*m[t] + B*W_in(x[t])   (W_in ← STDP + WTA)
        y population : y[t]   = C*m[t] + D*x[t]           (D    ← STDP + WTA)

    Both W_in [24→250] and D [24→250] receive raw spk_input (24-dim).
    WTA is applied to spk_y using curr_y (membrane potential) argmax —
    not spk_y argmax, which would always pick the lowest-index active neuron.
    """

    # STDP hyperparameters for D
    STDP_A_PLUS_D   = 0.02
    STDP_A_MINUS_D  = 0.01
    STDP_TAU_PRE_D  = 0.95
    STDP_TAU_POST_D = 0.95

    # STDP hyperparameters for W_in — asymmetric to break symmetry
    STDP_A_PLUS_WIN   = 0.01
    STDP_A_MINUS_WIN  = 0.005
    STDP_TAU_PRE_WIN  = 0.80
    STDP_TAU_POST_WIN = 0.80

    W_MAX = 1.0
    W_MIN = 0.0

    def __init__(self, input_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=1,          # dummy — not used in Exp1
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()

        # ── W_in [24→250]: trained by STDP + WTA (same strategy as D) ───
        self.W_in = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.W_in.weight, 0.0, 1.0)
        self.W_in.weight.requires_grad_(False)   # updated by STDP, not optimizer

        # ── D [24→250]: trained by STDP + WTA ────────────────────────────
        self.D = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.D.weight, 0.0, 1.0)
        self.D.weight.requires_grad_(False)      # updated by STDP, not optimizer

        # W_out: not used in Exp1
        self.W_out.weight.requires_grad_(False)

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"Neuron type '{neuron_type}' not found in snntorch.")

        self.spk_m = Neuron(
            beta=params['beta_spk_m'],
            threshold=params['threshold_spk_m'],
            learn_beta=False, learn_threshold=False, init_hidden=True,
        )
        self.spk_y = Neuron(
            beta=params['beta_spk_y'],
            threshold=params['threshold_spk_y'],
            learn_beta=False, learn_threshold=False, init_hidden=True,
        )

        # STDP traces for both W_in and D
        self.trace_pre_D    = None
        self.trace_post_D   = None
        self.trace_pre_Win  = None
        self.trace_post_Win = None

        # Spike history accumulated per batch
        self._spk_input_hist  = []   # raw spk_input — pre-synaptic for both
        self._spk_y_wta_hist  = []   # WTA-masked spk_y — post-synaptic for both

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def init_cell(self):
        """Reset LIF states and STDP traces. Call at the start of each batch."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.trace_pre_D    = None
        self.trace_post_D   = None
        self.trace_pre_Win  = None
        self.trace_post_Win = None
        self._spk_input_hist = []
        self._spk_y_wta_hist = []
        return torch.empty(0)

    def _init_traces(self, batch_size, device):
        self.trace_pre_D    = torch.zeros(batch_size, self.input_size,  device=device)
        self.trace_post_D   = torch.zeros(batch_size, self.memory_size, device=device)
        self.trace_pre_Win  = torch.zeros(batch_size, self.input_size,  device=device)
        self.trace_post_Win = torch.zeros(batch_size, self.memory_size, device=device)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        One-timestep forward pass.

        Returns:
            spk_y      : [B, memory_size]  raw y spikes (pre-WTA, for monitoring)
            spk_memory : [B, memory_size, order]
        """
        B, device = spk_input.shape[0], spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (B, self.memory_size, self.order), dtype=torch.float, device=device)

        if self.trace_pre_D is None:
            self._init_traces(B, device)

        # m population: W_in encodes raw input for state memory
        u_t        = self.W_in(spk_input)             # [B, 250]
        curr_m     = self.A(spk_memory) + self.B(u_t.unsqueeze(-1))
        spk_memory = self.spk_m(curr_m)

        # y population: D encodes raw input for learned representations
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        spk_y  = self.spk_y(curr_y)                   # [B, 250]

        # ── WTA: only the most excited y neuron updates weights ───────────
        # curr_y argmax (membrane potential) gives a clear winner.
        # spk_y_wta is used as post-synaptic signal for BOTH W_in and D.
        winner    = curr_y.argmax(dim=1, keepdim=True)   # [B, 1]
        wta_mask  = torch.zeros_like(spk_y)
        wta_mask.scatter_(1, winner, 1.0)
        spk_y_wta = spk_y * wta_mask                     # [B, 250]
        # ─────────────────────────────────────────────────────────────────

        # Save for STDP — same WTA signal used for both W_in and D
        self._spk_input_hist.append(spk_input.detach())
        self._spk_y_wta_hist.append(spk_y_wta.detach())

        # Return raw spk_y for monitoring
        return spk_y, spk_memory

    # ── STDP update ───────────────────────────────────────────────────────

    def stdp_update(self):
        """
        Weight-dependent STDP for both W_in and D, using WTA-masked
        post-synaptic spikes.

        Both matrices use the same WTA winner as post-synaptic signal.
        This ensures both W_in and D neurons specialize together toward
        the same distinct Braille patterns — as Prof. Fra requested.

        Diehl & Cook weight-dependent rule:
            dW = A_plus  * (W_MAX - w) * trace_pre  * spk_post_wta
               - A_minus * (w - W_MIN) * trace_post * spk_pre
        """
        if not self._spk_input_hist:
            return

        with torch.no_grad():
            w_D   = self.D.weight     # [250, 24]
            w_Win = self.W_in.weight  # [250, 24]

            for spk_input, spk_y_wta in zip(
                self._spk_input_hist,
                self._spk_y_wta_hist,
            ):
                # ── W_in update ───────────────────────────────────────────
                self.trace_pre_Win  = (
                    self.STDP_TAU_PRE_WIN  * self.trace_pre_Win  + spk_input
                )
                self.trace_post_Win = (
                    self.STDP_TAU_POST_WIN * self.trace_post_Win + spk_y_wta
                )
                dW_plus_Win  = self.STDP_A_PLUS_WIN * torch.einsum(
                    'bi,bj->ij', spk_y_wta, self.trace_pre_Win
                ) / spk_input.shape[0]
                dW_minus_Win = self.STDP_A_MINUS_WIN * torch.einsum(
                    'bi,bj->ij', self.trace_post_Win, spk_input
                ) / spk_input.shape[0]
                dW_Win = dW_plus_Win * (self.W_MAX - w_Win) - dW_minus_Win * (w_Win - self.W_MIN)
                self.W_in.weight.add_(dW_Win)

                # ── D update ──────────────────────────────────────────────
                self.trace_pre_D  = (
                    self.STDP_TAU_PRE_D  * self.trace_pre_D  + spk_input
                )
                self.trace_post_D = (
                    self.STDP_TAU_POST_D * self.trace_post_D + spk_y_wta
                )
                dW_plus_D  = self.STDP_A_PLUS_D * torch.einsum(
                    'bi,bj->ij', spk_y_wta, self.trace_pre_D
                ) / spk_input.shape[0]
                dW_minus_D = self.STDP_A_MINUS_D * torch.einsum(
                    'bi,bj->ij', self.trace_post_D, spk_input
                ) / spk_input.shape[0]
                dW_D = dW_plus_D * (self.W_MAX - w_D) - dW_minus_D * (w_D - self.W_MIN)
                self.D.weight.add_(dW_D)

            # Clip both to [W_MIN, W_MAX]
            self.W_in.weight.clamp_(self.W_MIN, self.W_MAX)
            self.D.weight.clamp_(self.W_MIN, self.W_MAX)

        self._spk_input_hist = []
        self._spk_y_wta_hist = []