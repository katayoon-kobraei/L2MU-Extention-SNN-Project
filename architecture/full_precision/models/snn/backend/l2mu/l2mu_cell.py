import snntorch
import torch
import torch.nn as nn
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Experiment 2B: SSM core with m, y, and out populations.
    D is frozen (loaded from Exp1).
    W_out (54 neurons = 2 per class x 27 classes) trained with supervised
    rate-code S2-STDP.

    Architecture:
        m population: m[t+1] = A*m[t] + B*e_x(x[t])      (fixed)
        y population: y[t]   = C*m[t] + D*x[t]            (D frozen from Exp1)
        out population: o[t] = W_out * spk_y[t]            (W_out <- S2-STDP)

    Rate-code S2-STDP
    -----------------
    rate_pre_i  = sum_t(spk_y_i)   / T   in [0,1]
    rate_post_j = sum_t(spk_out_j) / T   in [0,1]

    Fixed absolute desired rates (NOT mean-relative):
        target neuron     (c_j == y, role=TARGET):    r*_j = R_TARGET
        non-target neuron (c_j != y | role=NON_TARGET): r*_j = R_NON_TARGET

    Using fixed targets instead of R_mean +/- g prevents the oscillation
    that occurs when the population mean swings wildly: mean-relative targets
    move with the collapse/surge, amplifying it. Fixed targets are a stable
    attractor regardless of current population state.

    Error:
        e_j = rate_post_j - r*_j
            > 0  -> neuron too active   -> depress   incoming weights
            < 0  -> neuron too inactive -> potentiate incoming weights

    Weight update (Hebbian + rescue):
        dW[j,i] = -e_j * (A_LR * rate_pre_i + A_LR_ERR)

        Term A: A_LR * rate_pre_i  -- Hebbian locality
        Term B: A_LR_ERR           -- error-only rescue for dead neurons
            When a neuron is silent (rate_post=0, e_j<0, rate_pre~0),
            term B provides a non-zero potentiation signal so it can recover.

    Paired neuron scheme (Section 4.3, NeurIPS 2024):
        index 2*c   -> TARGET     neuron for class c (learns to fire more)
        index 2*c+1 -> NON_TARGET neuron for class c (always gets non-target error)
    """

    # S2-STDP hyperparameters
    # Learning rate calibration:
    #   per-synapse update ~ A_LR * |e| * rate_pre ~ 0.1 * 0.1 * 0.17 = 0.0017/epoch
    #   This moves W_std by ~0.002/epoch -> reaches ~0.1 difference after ~50 epochs.
    S2STDP_A_LR     = 1.0    # Hebbian learning rate
    S2STDP_A_LR_ERR = 0.02  # error-only rescue learning rate

    # Fixed absolute desired firing rates
    # Calibrated from sanity check: y_rate~0.17, init out_rate~0.08
    # target wants more than background, non-target wants near-silence
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

        self.e_x.weight.requires_grad_(False)
        self.D.weight.requires_grad_(False)
        self.W_out.weight.requires_grad_(False)
        torch.nn.init.uniform_(self.W_out.weight, -0.1, 0.1)

        # Neuron metadata buffers (auto-move to GPU with .to(device))
        roles, cls_map = [], []
        for c in range(self.NUM_CLASSES):
            roles   += [self.ROLE_TARGET, self.ROLE_NON_TARGET]
            cls_map += [c, c]
        self.register_buffer('neuron_roles',
                             torch.tensor(roles,   dtype=torch.long))
        self.register_buffer('neuron_class',
                             torch.tensor(cls_map, dtype=torch.long))

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

        self._rate_pre  = None   # [B, memory_size]
        self._rate_post = None   # [B, NUM_OUT]
        self._T_steps   = 0

    def load_D_from_exp1(self, D_weights_path):
        D_weights = torch.load(D_weights_path, map_location='cpu')
        self.D.weight.data.copy_(D_weights)
        self.D.weight.requires_grad_(False)
        print(f"Loaded D weights from {D_weights_path}")
        print(f"D weight mean: {self.D.weight.mean().item():.4f}  "
              f"std: {self.D.weight.std().item():.4f}")

    def init_cell(self):
        """Reset LIF states and accumulators. Call once per batch."""
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

        u_t        = self.e_x(spk_input)
        curr_m     = self.A(spk_memory) + self.B(u_t.unsqueeze(-1))
        spk_memory = self.spk_m(curr_m)

        curr_y = self.C(spk_memory).squeeze(-1) + self.D(u_t)
        spk_y  = self.spk_y(curr_y)

        curr_out = self.W_out(spk_y)
        spk_out  = self.spk_out(curr_out)

        if self._rate_pre is None:
            self._rate_pre  = spk_y.detach().clone()
            self._rate_post = spk_out.detach().clone()
        else:
            self._rate_pre  += spk_y.detach()
            self._rate_post += spk_out.detach()
        self._T_steps += 1

        return spk_out, spk_y, spk_memory

    def s2stdp_update(self, labels: torch.Tensor):
        """
        Rate-code S2-STDP weight update for W_out.
        Call once per batch, after the complete T-step forward pass.

        Args:
            labels : [B] -- integer class labels 0..26
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

            # Error: positive -> too active -> depress
            #        negative -> too silent -> potentiate
            errors = rate_post - r_desired   # [B, NUM_OUT]

            # Weight update:
            #   dW[j,i] = -e_j * (A_LR * rate_pre_i + A_LR_ERR)
            # errors:   [B, NUM_OUT, 1]
            # factor:   [B, 1, memory_size]  -> broadcasts to [B, NUM_OUT, memory_size]
            factor = (
                self.S2STDP_A_LR * rate_pre.unsqueeze(1)
                + self.S2STDP_A_LR_ERR
            )  # [B, 1, memory_size]

            dW = -(errors.unsqueeze(2) * factor)   # [B, NUM_OUT, memory_size]
            dW = dW.mean(dim=0)                    # [NUM_OUT, memory_size]

            self.W_out.weight.add_(dW)
            self.W_out.weight.clamp_(self.W_MIN, self.W_MAX)

        self._rate_pre  = None
        self._rate_post = None
        self._T_steps   = 0