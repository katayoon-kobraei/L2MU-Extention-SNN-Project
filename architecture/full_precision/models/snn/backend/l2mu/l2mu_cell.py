import snntorch
import torch
import torch.nn as nn
from architecture.full_precision.core.lmu.interface import LMUCore
from architecture.full_precision.core.lmu.utils import XavierLinear


class L2MUCell(LMUCore):
    """
    Experiment 2C: Exp2B + WTA on the y population.

    Key change from Exp2B:
        After spk_y is computed, only the neuron with the highest
        activation per sample passes through (Winner-Take-All mask).
        This forces each y neuron to specialize toward a different subset
        of Braille patterns — the competitive pressure that S2-STDP alone
        cannot create.

    Architecture:
        m population : m[t+1] = A*m[t] + B*W_in(x[t])    (W_in frozen from Exp1)
        y population : y[t]   = C*m[t] + D*x[t]           (D    frozen from Exp1)
        WTA          : spk_y_wta = argmax mask on spk_y    ← NEW
        out population: o[t]  = W_out * spk_y_wta          (W_out ← S2-STDP)

    W_in [250, 24] and D [250, 24] both receive raw spk_input (24-dim),
    exactly matching how they were trained in Exp1.

    Changes vs l2mu_cell_exp2b.py
    ─────────────────────────────
    __init__ :  override D → [input_size, memory_size] = [24→250]
                add       W_in [24→250] (replaces e_x)
                add       load_W_in_from_exp1()
    forward  :  u_t  = W_in(spk_input)          (was e_x)
                D    receives spk_input           (was u_t)
                WTA block after spk_y             ← NEW
                W_out receives spk_y_wta          (was spk_y)
                rate_pre accumulated from spk_y_wta (was spk_y)
    s2stdp_update: unchanged
    """

    # ── S2-STDP hyperparameters — identical to best Exp2B run ─────────────
    S2STDP_A_LR     = 1.0
    S2STDP_A_LR_ERR = 0.02

    R_TARGET     = 0.15
    R_NON_TARGET = 0.02

    W_MAX =  1.0
    W_MIN = -1.0

    NUM_CLASSES     = 27
    NEURONS_PER_CLS = 2
    NUM_OUT         = NUM_CLASSES * NEURONS_PER_CLS   # 54

    ROLE_TARGET     = 0
    ROLE_NON_TARGET = 1

    def __init__(self, input_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=self.NUM_OUT,
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()

        # ── Fix 1: override D to [input_size, memory_size] = [24, 250] ──
        # The base interface (exp2b) creates D[250,250].
        # Exp1 saved D[250,24] — both receive raw 24-dim spk_input.
        # We replace it here so load_D_from_exp1 copies cleanly.
        self.D = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.D.weight, 0.0, 1.0)
        self.D.weight.requires_grad_(False)

        # ── Fix 2: add W_in [input_size, memory_size] = [24, 250] ───────
        # The base interface (exp2b) has e_x (LeCun, random, fixed).
        # Exp1 saved W_in[250,24] — we load it so the m population sees
        # the same u_t it was trained with.
        self.W_in = XavierLinear(self.input_size, self.memory_size, bias=False)
        torch.nn.init.uniform_(self.W_in.weight, 0.0, 1.0)
        self.W_in.weight.requires_grad_(False)
        # (self.e_x from the base interface still exists but is unused)

        # ── W_out: 54 neurons, trained by S2-STDP ────────────────────────
        self.W_out.weight.requires_grad_(False)
        torch.nn.init.uniform_(self.W_out.weight, -0.1, 0.1)

        # Neuron metadata (auto-moves to GPU with .to(device))
        roles, cls_map = [], []
        for c in range(self.NUM_CLASSES):
            roles   += [self.ROLE_TARGET, self.ROLE_NON_TARGET]
            cls_map += [c, c]
        self.register_buffer('neuron_roles', torch.tensor(roles,   dtype=torch.long))
        self.register_buffer('neuron_class', torch.tensor(cls_map, dtype=torch.long))

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
        self.spk_out = Neuron(
            beta=params['beta_spk_out'], threshold=params['threshold_spk_out'],
            learn_beta=False, learn_threshold=False, init_hidden=True,
        )

        self._rate_pre  = None   # [B, memory_size] — accumulated from spk_y_wta
        self._rate_post = None   # [B, NUM_OUT]
        self._T_steps   = 0

    # ── Weight loading ────────────────────────────────────────────────────

    def load_D_from_exp1(self, D_weights_path):
        """Load D [250,24] from Exp1 and freeze."""
        D_weights = torch.load(D_weights_path, map_location='cpu')
        self.D.weight.data.copy_(D_weights)
        self.D.weight.requires_grad_(False)
        print(f"Loaded D     : shape={tuple(self.D.weight.shape)}  "
              f"mean={self.D.weight.mean():.4f}  std={self.D.weight.std():.4f}")

    def load_W_in_from_exp1(self, W_in_weights_path):
        """Load W_in [250,24] from Exp1 and freeze."""
        W_in_weights = torch.load(W_in_weights_path, map_location='cpu')
        self.W_in.weight.data.copy_(W_in_weights)
        self.W_in.weight.requires_grad_(False)
        print(f"Loaded W_in  : shape={tuple(self.W_in.weight.shape)}  "
              f"mean={self.W_in.weight.mean():.4f}  std={self.W_in.weight.std():.4f}")

    # ── Core methods ──────────────────────────────────────────────────────

    def init_cell(self):
        """Reset LIF states and rate accumulators. Call once per batch."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        self._rate_pre  = None
        self._rate_post = None
        self._T_steps   = 0
        return torch.empty(0)

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        One-timestep forward pass.

        Args:
            spk_input  : [B, input_size]         binary spike vector
            spk_memory : [B, memory_size, order]  (empty tensor on first call)
        Returns:
            spk_out    : [B, NUM_OUT]             output population spikes
            spk_y      : [B, memory_size]         raw y spikes (pre-WTA, for monitoring)
            spk_memory : [B, memory_size, order]  updated state
        """
        B, device = spk_input.shape[0], spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (B, self.memory_size, self.order), dtype=torch.float, device=device)

        # ── m population: W_in encodes raw input for state memory ─────────
        u_t        = self.W_in(spk_input)              # [B, 250]  ← was e_x
        curr_m     = self.A(spk_memory) + self.B(u_t.unsqueeze(-1))
        spk_memory = self.spk_m(curr_m)

        # ── y population: D encodes raw input for learned representations ──
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)  # D: raw ← was D(u_t)
        spk_y  = self.spk_y(curr_y)                    # [B, 250]

        # ── WTA: only the winning y neuron passes to W_out ───────────────
        winner    = curr_y.argmax(dim=1, keepdim=True)   # membrane potential, not binary spikes
        wta_mask  = torch.zeros_like(spk_y)
        wta_mask.scatter_(1, winner, 1.0)
        spk_y_wta = spk_y * wta_mask                   # [B, 250] — at most 1 active per sample
        # ─────────────────────────────────────────────────────────────────

        # ── out population: receives WTA-gated y spikes ───────────────────
        curr_out = self.W_out(spk_y_wta)               # [B, 54]  ← was W_out(spk_y)
        spk_out  = self.spk_out(curr_out)              # [B, 54]

        # Accumulate rates — pre-synaptic signal for W_out is spk_y_wta
        if self._rate_pre is None:
            self._rate_pre  = spk_y_wta.detach().clone()   # ← was spk_y
            self._rate_post = spk_out.detach().clone()
        else:
            self._rate_pre  += spk_y_wta.detach()          # ← was spk_y
            self._rate_post += spk_out.detach()
        self._T_steps += 1

        # Return raw (pre-WTA) spk_y so the train loop can monitor y rates
        return spk_out, spk_y, spk_memory

    def s2stdp_update(self, labels: torch.Tensor):
        """
        Rate-code S2-STDP weight update for W_out.
        Identical to Exp2B — no changes needed here because _rate_pre
        is now accumulated from spk_y_wta in forward().

        Call once per batch, after the complete T-step forward pass.
        """
        if self._rate_pre is None or self._T_steps == 0:
            return

        with torch.no_grad():
            T   = float(self._T_steps)
            dev = labels.device

            rate_pre  = self._rate_pre  / T   # [B, memory_size]
            rate_post = self._rate_post / T   # [B, NUM_OUT]

            # Fixed desired rates: non-target for all neurons by default
            r_desired = torch.full(
                (rate_post.shape[0], self.NUM_OUT),
                self.R_NON_TARGET, dtype=torch.float, device=dev,
            )

            # Target neuron of the sample's class gets R_TARGET
            tgt_neuron_idx = labels * self.NEURONS_PER_CLS   # [B]
            batch_idx      = torch.arange(labels.shape[0], device=dev)
            r_desired[batch_idx, tgt_neuron_idx] = self.R_TARGET

            # Error: positive → too active → depress
            #        negative → too silent → potentiate
            errors = rate_post - r_desired   # [B, NUM_OUT]

            # dW[j,i] = -e_j * (A_LR * rate_pre_i + A_LR_ERR)
            factor = (
                self.S2STDP_A_LR * rate_pre.unsqueeze(1)   # [B, 1, memory_size]
                + self.S2STDP_A_LR_ERR
            )

            dW = -(errors.unsqueeze(2) * factor)   # [B, NUM_OUT, memory_size]
            dW = dW.mean(dim=0)                    # [NUM_OUT, memory_size]

            self.W_out.weight.add_(dW)
            self.W_out.weight.clamp_(self.W_MIN, self.W_MAX)

        self._rate_pre  = None
        self._rate_post = None
        self._T_steps   = 0