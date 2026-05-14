import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Experiment 1: SSM core with ONLY m and y populations.
    No out population, no W_out, no classification.

    Goal: train D with unsupervised STDP so that the y population
    learns to represent prototypical Braille letter patterns,
    purely from spike correlations — no labels used at all.

    Architecture:
        m population: m[t+1] = A * m[t] + B * x[t]   (A, B fixed)
        y population: y[t]   = C * m[t] + D * x[t]   (C fixed, D STDP)

    Fixes vs previous version:
        1. Removed e_x — B now receives spk_input directly summed
           to one scalar per memory slot, no random projection
        2. STDP traces reset between EVERY sample (not just per batch)
           following Diehl & Cook 2015 rest phase between samples
        3. Weight-dependent STDP update — weights near W_MAX get
           smaller potentiation, weights near W_MIN get smaller
           depression — creates natural stabilization
    """

    STDP_A_PLUS   = 0.02
    STDP_A_MINUS  = 0.02
    STDP_TAU_PRE  = 0.95
    STDP_TAU_POST = 0.95

    W_MAX = 1.0
    W_MIN = 0.0

    def __init__(self, input_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=1,        # dummy — not used in exp1
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()

        # D trained by STDP only — excluded from any optimizer
        self.D.weight.requires_grad_(False)

        # W_out not needed in Exp1
        self.W_out.weight.requires_grad_(False)

        # e_x is a fixed random projection — applied once to x[t] before B and D
        # not trained, just a fixed input encoder
        self.e_x.weight.requires_grad_(False)

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

        # STDP traces — reset per sample, not per batch
        self.trace_pre_D  = None
        self.trace_post_D = None

        # Spike history per timestep — saved during forward
        self._spk_input_hist = []
        self._spk_y_hist     = []

    def init_cell(self):
        """
        Reset LIF states and STDP traces.
        Called at the start of EACH sample — following Diehl & Cook
        rest phase between samples where all variables decay to rest.
        """
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self._reset_traces_and_history()
        return torch.empty(0)   # spk_memory placeholder

    def _reset_traces_and_history(self):
        """Reset STDP traces and spike history — called between samples."""
        self.trace_pre_D  = None
        self.trace_post_D = None
        self._spk_input_hist = []
        self._spk_y_hist     = []

    def _init_traces(self, batch_size, device):
        # trace_pre_D tracks the pre-synaptic activity of D.
        # Since D now receives u_t = e_x(spk_input) instead of raw spk_input,
        # the pre-synaptic dimension changes from input_size (24) to memory_size (250).
        # The trace shape must match D's input dimension for the STDP einsum to work.
        self.trace_pre_D = torch.zeros(batch_size, self.memory_size, device=device)
        self.trace_post_D = torch.zeros(batch_size, self.memory_size, device=device)

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        Forward pass — m and y populations only.
        e_x applied once at input boundary, u_t shared between B and D.
        """
        batch_size = spk_input.shape[0]
        device     = spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (batch_size, self.memory_size, self.order),
                dtype=torch.float, device=device)

        if self.trace_pre_D is None:
            self._init_traces(batch_size, device)

        # e_x is applied ONCE at the input boundary, projecting x[t] from input_size (24)
        # to memory_size (250). The result u_t is shared between both B and D paths.
        # This replaces the previous mean pooling (which gave all m neurons the same scalar)
        # with a fixed random projection that gives each neuron a different linear combination
        # of the 24 input channels, enabling proper specialization in both populations.

        u_t    = self.e_x(spk_input)              # [B, memory_size] — projected once
        u_t_3d = u_t.unsqueeze(-1)               # [B, memory_size, 1]
        curr_m = self.A(spk_memory) + self.B(u_t_3d)
        spk_memory = self.spk_m(curr_m)                 # [B, memory_size, order]

        # y population: y[t] = C * m[t] + D * x[t]
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(u_t)   # D gets u_t not spk_input
        spk_y  = self.spk_y(curr_y)                     # [B, memory_size]

        # Save for STDP
        self._spk_input_hist.append(u_t.detach())  # save projected input
        self._spk_y_hist.append(spk_y.detach())

        return spk_y, spk_memory

    def stdp_update(self):
        """
        Apply weight-dependent unsupervised STDP to D.
        Call this after processing each sample (not batch).

        Weight-dependent rule (Diehl & Cook 2015):
            dW = A_plus  * (W_MAX - w) * trace_pre  * spk_post
               - A_minus * (w - W_MIN) * trace_post * spk_pre

        Weights near W_MAX: potentiation is small → natural ceiling
        Weights near W_MIN: depression is small  → natural floor
        This creates stable diverse weight patterns.
        """
        if not self._spk_input_hist:
            return

        with torch.no_grad():
            w = self.D.weight    # [memory_size, memory_size] — D: u_t (250) → y (250)

            for spk_input, spk_y in zip(
                self._spk_input_hist,
                self._spk_y_hist,
            ):
                # Update traces
                self.trace_pre_D  = (
                    self.STDP_TAU_PRE  * self.trace_pre_D  + spk_input
                )    # [B, memory_size] — pre trace over u_t dimension
                self.trace_post_D = (
                    self.STDP_TAU_POST * self.trace_post_D + spk_y
                )   # [memory_size, memory_size] — dW shape matches D

                # Potentiation: post fires → strengthen from active pre
                # Weight-dependent: scaled by (W_MAX - w)
                dW_plus = self.STDP_A_PLUS * torch.einsum(
                    'bi,bj->ij', spk_y, self.trace_pre_D
                ) / spk_input.shape[0]   # [memory_size, input_size]

                # Depression: pre fires → weaken toward active post
                # Weight-dependent: scaled by (w - W_MIN)
                dW_minus = self.STDP_A_MINUS * torch.einsum(
                    'bi,bj->ij', self.trace_post_D, spk_input
                ) / spk_input.shape[0]  # [memory_size, input_size]

                # Weight-dependent update
                dW = dW_plus * (self.W_MAX - w) - dW_minus * (w - self.W_MIN)
                self.D.weight.add_(dW)

            # Clip after all timesteps
            self.D.weight.clamp_(self.W_MIN, self.W_MAX)

        # Clear history for next sample
        self._spk_input_hist = []
        self._spk_y_hist     = []
