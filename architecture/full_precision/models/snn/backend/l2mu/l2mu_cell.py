import snntorch
import torch
import torch.nn as nn
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Experiment 2B: SSM core with m, y, and out populations.
    D is frozen (loaded from Exp1).
    W_out (54 neurons = 2 per class × 27 classes) trained with supervised
    rate-code S2-STDP.

    Architecture:
        m population: m[t+1] = A·m[t] + B·e_x(x[t])      (fixed)
        y population: y[t]   = C·m[t] + D·x[t]            (D frozen from Exp1)
        out population: o[t] = W_out · spk_y[t]            (W_out ← S2-STDP)

    ── Rate-code S2-STDP ───────────────────────────────────────────────────
    Notation (per sample, per batch):
        rate_pre_i   = sum_t(spk_y_i) / T          ∈ [0,1]  pre-synaptic rate
        rate_post_j  = sum_t(spk_out_j) / T        ∈ [0,1]  post-synaptic rate
        R_mean       = mean_j(rate_post_j)                   mean output rate

    Desired rates (adapted from S2-STDP Eq. 5, Goupy et al. 2024):
        target neuron     (c_j == y, role=TARGET):
            r*_j = R_mean + (C-1)/C · g        ← push above average
        non-target neuron (c_j != y OR role=NON_TARGET):
            r*_j = R_mean - 1/C · g            ← push below average

    Error:
        e_j = rate_post_j - r*_j
            > 0  → neuron is too active   → depress   incoming weights
            < 0  → neuron is too inactive → potentiate incoming weights

    Weight update — two complementary terms:

      Term A (Hebbian, locality):
          dW[j,i] = -e_j · A_LR · rate_pre_i
          Updates only synapses that are actually co-active.
          Problem: if neuron j is dead (rate_post=0), error e_j < 0 but
          rate_pre might also be near 0 for many i → update ≈ 0 → stuck.

      Term B (error-only rescue):
          dW[j,i] = -e_j · A_LR_ERR
          A small uniform update proportional only to the error.
          This rescues dead neurons: even when rate_pre ≈ 0, a negative e_j
          will push all incoming weights slightly positive, eventually making
          the neuron cross threshold and start firing.

    Combined:
        dW[j,i] = -e_j · (A_LR · rate_pre_i + A_LR_ERR)

    ── Paired neuron scheme (Section 4.3, NeurIPS 2024 paper) ──────────────
        - 2 neurons per class c:
            index 2·c   → TARGET     neuron (learns to fire MORE for class c)
            index 2·c+1 → NON_TARGET neuron (always gets non-target error)
        - The non-target neuron's role overrides the sample label:
          even when the sample label matches its class, it is treated as
          non-target, pushing it to specialise toward what class c is NOT.
    """

    # ── S2-STDP hyperparameters ──────────────────────────────────────────
    # A_LR: Hebbian term learning rate.  Scale: e~0.1, rate_pre~0.17 →
    #   per-synapse update ~ 0.1 × 1.0 × 0.17 = 0.017/epoch → W_std moves.
    S2STDP_A_LR      = 1.0    # Hebbian learning rate (was 0.01 — 100× too small)

    # A_LR_ERR: error-only term, provides dead-neuron rescue.
    # Much smaller than A_LR to avoid washing out the Hebbian locality.
    S2STDP_A_LR_ERR  = 0.05   # error-only learning rate

    # g: desired rate spread fraction.
    # With g=0.1 and 27 classes:
    #   target   desired = R_mean + (26/27)·0.1 ≈ R_mean + 0.096
    #   nontarget desired = R_mean - (1/27)·0.1 ≈ R_mean - 0.004
    # This creates a strong push on target neurons and a gentle push on
    # non-target neurons — appropriate when output rates are low.
    S2STDP_G         = 0.1

    W_MAX =  1.0
    W_MIN = -1.0

    NUM_CLASSES      = 27
    NEURONS_PER_CLS  = 2
    NUM_OUT          = NUM_CLASSES * NEURONS_PER_CLS   # 54

    ROLE_TARGET      = 0
    ROLE_NON_TARGET  = 1

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

        # ── Fixed / frozen weights ────────────────────────────────────────
        self.e_x.weight.requires_grad_(False)
        self.D.weight.requires_grad_(False)

        # ── W_out: trained exclusively by S2-STDP ────────────────────────
        self.W_out.weight.requires_grad_(False)
        # Small symmetric init — equal chance of going + or −
        torch.nn.init.uniform_(self.W_out.weight, -0.1, 0.1)

        # ── Neuron metadata (registered buffers → follow .to(device)) ────
        roles, cls_map = [], []
        for c in range(self.NUM_CLASSES):
            roles  += [self.ROLE_TARGET, self.ROLE_NON_TARGET]
            cls_map += [c, c]
        self.register_buffer('neuron_roles',
                             torch.tensor(roles,   dtype=torch.long))
        self.register_buffer('neuron_class',
                             torch.tensor(cls_map, dtype=torch.long))

        # ── LIF neurons ──────────────────────────────────────────────────
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

        # ── Accumulators (reset per sample in init_cell) ─────────────────
        self._rate_pre  = None   # [B, memory_size]
        self._rate_post = None   # [B, NUM_OUT]
        self._T_steps   = 0

    # ─────────────────────────────────────────────────────────────────────

    def load_D_from_exp1(self, D_weights_path):
        D_weights = torch.load(D_weights_path, map_location='cpu')
        self.D.weight.data.copy_(D_weights)
        self.D.weight.requires_grad_(False)
        print(f"Loaded D weights from {D_weights_path}")
        print(f"D weight mean: {self.D.weight.mean().item():.4f}  "
              f"std: {self.D.weight.std().item():.4f}")

    def init_cell(self):
        """Reset LIF states and spike accumulators. Call once per batch."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        self._rate_pre  = None
        self._rate_post = None
        self._T_steps   = 0
        return torch.empty(0)

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        One-timestep forward pass.

        Args:
            spk_input  : [B, input_size]
            spk_memory : [B, memory_size, order]  (empty tensor on first call)
        Returns:
            spk_out    : [B, NUM_OUT]
            spk_y      : [B, memory_size]
            spk_memory : [B, memory_size, order]
        """
        B, device = spk_input.shape[0], spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (B, self.memory_size, self.order), dtype=torch.float, device=device)

        # m population
        u_t    = self.e_x(spk_input)
        curr_m = self.A(spk_memory) + self.B(u_t.unsqueeze(-1))
        spk_memory = self.spk_m(curr_m)

        # y population (D frozen from Exp1)
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        spk_y  = self.spk_y(curr_y)         # [B, memory_size]

        # out population (W_out trained by S2-STDP)
        curr_out = self.W_out(spk_y)        # [B, NUM_OUT]
        spk_out  = self.spk_out(curr_out)   # [B, NUM_OUT]

        # Accumulate spike counts for S2-STDP
        if self._rate_pre is None:
            self._rate_pre  = spk_y.detach().clone()
            self._rate_post = spk_out.detach().clone()
        else:
            self._rate_pre  += spk_y.detach()
            self._rate_post += spk_out.detach()
        self._T_steps += 1

        return spk_out, spk_y, spk_memory

    # ─────────────────────────────────────────────────────────────────────

    def s2stdp_update(self, labels: torch.Tensor):
        """
        Rate-code S2-STDP weight update for W_out.
        Call once per batch, AFTER the complete T-step forward pass.

        Args:
            labels : [B]  — integer class labels 0..26
        """
        if self._rate_pre is None or self._T_steps == 0:
            return

        with torch.no_grad():
            T   = float(self._T_steps)
            C   = self.NUM_CLASSES
            g   = self.S2STDP_G
            dev = labels.device

            # ── Firing rates ──────────────────────────────────────────────
            rate_pre  = self._rate_pre  / T   # [B, memory_size]  ∈ [0,1]
            rate_post = self._rate_post / T   # [B, NUM_OUT]       ∈ [0,1]

            # ── Mean output rate per sample ───────────────────────────────
            R_mean = rate_post.mean(dim=1, keepdim=True)   # [B, 1]

            # ── Desired rates ─────────────────────────────────────────────
            r_target     = (R_mean + (C - 1) / C * g).clamp(0.0, 1.0)  # [B,1]
            r_non_target = (R_mean - 1.0 / C * g).clamp(0.0, 1.0)      # [B,1]

            # ── Build desired-rate tensor [B, NUM_OUT] ─────────────────────
            # Default: every neuron gets the non-target desired rate
            r_desired = r_non_target.expand(-1, self.NUM_OUT).clone()

            # Override: for each sample b, the TARGET neuron of its class
            # gets the target desired rate.
            # target neuron index for class c = 2*c  (vectorised, no Python loop)
            tgt_neuron_idx = labels * self.NEURONS_PER_CLS   # [B]
            batch_idx      = torch.arange(labels.shape[0], device=dev)
            r_desired[batch_idx, tgt_neuron_idx] = r_target[batch_idx, 0]

            # ── Error per output neuron ────────────────────────────────────
            # e_j > 0 → too active   → depress
            # e_j < 0 → too inactive → potentiate
            errors = rate_post - r_desired   # [B, NUM_OUT]

            # ── Combined weight update ─────────────────────────────────────
            # Term A (Hebbian locality):
            #   contribution[b,j,i] = -e_j[b] · A_LR · rate_pre_i[b]
            # Term B (error-only rescue for dead neurons):
            #   contribution[b,j,i] = -e_j[b] · A_LR_ERR
            #
            # Together:
            #   dW[j,i] = mean_b { -e_j[b] · (A_LR · rate_pre_i[b] + A_LR_ERR) }
            #
            # Shapes:
            #   errors           : [B, NUM_OUT]
            #   rate_pre         : [B, memory_size]
            #   errors.unsqueeze : [B, NUM_OUT, 1]
            #   rate_pre.unqueeze: [B, 1, memory_size]

            hebbian_factor = (
                self.S2STDP_A_LR     * rate_pre.unsqueeze(1)                  # [B, 1, mem]
                + self.S2STDP_A_LR_ERR * torch.ones(1, 1, self.memory_size,   # [1, 1, mem]
                                                     device=dev)
            )  # [B, 1, memory_size]  — broadcast over NUM_OUT in next step

            dW = -(errors.unsqueeze(2) * hebbian_factor)   # [B, NUM_OUT, memory_size]
            dW = dW.mean(dim=0)                            # [NUM_OUT, memory_size]

            self.W_out.weight.add_(dW)
            self.W_out.weight.clamp_(self.W_MIN, self.W_MAX)

        # Reset accumulators
        self._rate_pre  = None
        self._rate_post = None
        self._T_steps   = 0