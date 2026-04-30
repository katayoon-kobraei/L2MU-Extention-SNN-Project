import snntorch
import torch
import torch.nn as nn
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):
    """
    Three-population SSM cell with STDP for D and W_out.

    Key design: STDP updates are NOT inside forward().
    Instead, forward() just saves the spike history,
    and stdp_update() is called separately after backward().
    This avoids any conflict with PyTorch autograd.
    """

    STDP_A_PLUS   = 0.01
    STDP_A_MINUS  = 0.01
    STDP_TAU_PRE  = 0.95
    STDP_TAU_POST = 0.95

    def __init__(self, input_size, output_size, params,
                 trainable_theta=False, neuron_type='Leaky'):
        super().__init__(
            input_size=input_size,
            memory_size=int(params['memory_size']),
            order=int(params['order']),
            theta=params['theta'],
            output_size=output_size,
            trainable_theta=trainable_theta,
            discretizer=params.get('discretizer', 'zoh'),
        )
        self.init_parameters()

        # D trained by STDP only — excluded from Adam
        # W_out trained by backprop — needs gradient path to the loss
        self.D.weight.requires_grad_(False)
        self.W_out.weight.requires_grad_(True)   # backprop

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"Neuron type '{neuron_type}' not found in snntorch.")

        self.spk_m = Neuron(beta=params['beta_spk_m'], threshold=params['threshold_spk_m'],
                            learn_beta=True, learn_threshold=True, init_hidden=True)
        self.spk_y = Neuron(beta=params['beta_spk_y'], threshold=params['threshold_spk_y'],
                            learn_beta=True, learn_threshold=True, init_hidden=True)
        self.spk_out = Neuron(beta=params['beta_spk_out'], threshold=params['threshold_spk_out'],
                              learn_beta=True, learn_threshold=True, init_hidden=True)

        # STDP traces for D only
        self.trace_pre_D  = None
        self.trace_post_D = None

        # Spike history for STDP — saved during forward, used in stdp_update()
        self._spk_input_hist = []
        self._spk_y_hist     = []

    def init_cell(self):
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        self.trace_pre_D  = None
        self.trace_post_D = None
        self._spk_input_hist = []
        self._spk_y_hist     = []
        return torch.empty(0)

    def _init_traces(self, batch_size, device):
        self.trace_pre_D  = torch.zeros(batch_size, self.input_size,  device=device)
        self.trace_post_D = torch.zeros(batch_size, self.memory_size, device=device)

    def forward(self, spk_input: torch.Tensor, spk_memory: torch.Tensor):
        """
        Forward pass — clean, no in-place weight modifications.
        Spike history is saved for STDP which runs separately after backward().
        """
        batch_size = spk_input.shape[0]
        device     = spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (batch_size, self.memory_size, self.order),
                dtype=torch.float, device=device)

        if self.trace_pre_D is None:
            self._init_traces(batch_size, device)

        # m population
        u_t    = self.e_x(spk_input)
        u_t_3d = u_t.unsqueeze(-1)
        curr_m = self.A(spk_memory) + self.B(u_t_3d)
        if self.discretizer == 'euler' and self.trainable_theta:
            curr_m = curr_m + curr_m * self.theta_inv
        spk_memory = self.spk_m(curr_m)

        # y population
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        spk_y  = self.spk_y(curr_y)

        # out population
        curr_out = self.W_out(spk_y)
        spk_out  = self.spk_out(curr_out)

        # Save spikes for STDP on D (detached — no gradient needed)
        if self.training:
            self._spk_input_hist.append(spk_input.detach())
            self._spk_y_hist.append(spk_y.detach())

        return spk_out, spk_memory

    def stdp_update(self):
        """
        Apply STDP to D only using the spike history saved during forward().
        W_out is now trained by backprop instead — it needs a gradient path to the loss.
        Call this AFTER loss.backward() and optimizer.step().
        """
        if not self._spk_input_hist:
            return  # nothing to update (e.g. during val/test)

        with torch.no_grad():
            for spk_input, spk_y in zip(
                self._spk_input_hist,
                self._spk_y_hist,
            ):
                # --- STDP for D only (pre=spk_input, post=spk_y) ---
                self.trace_pre_D  = self.STDP_TAU_PRE  * self.trace_pre_D  + spk_input
                self.trace_post_D = self.STDP_TAU_POST * self.trace_post_D + spk_y

                dW_plus  = self.STDP_A_PLUS  * torch.einsum('bi,bj->ij', spk_y, self.trace_pre_D)    / spk_input.shape[0]
                dW_minus = self.STDP_A_MINUS * torch.einsum('bi,bj->ij', self.trace_post_D, spk_input) / spk_input.shape[0]
                self.D.weight.add_(dW_plus - dW_minus)

        # Clear history for next batch
        self._spk_input_hist = []
        self._spk_y_hist     = []
        self._spk_out_hist   = []
