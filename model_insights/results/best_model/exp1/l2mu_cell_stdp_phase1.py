"""
l2mu_cell_stdp_phase1.py
========================
Phase 1: Unsupervised STDP training of the D matrix only.

What is trained / frozen
-------------------------
FROZEN  : A, B, C  (fixed LMU matrices — never change)
          W_out    (not touched at all in this phase)
          e_x      (input encoder — frozen so the SSM stays stable)
          spk_m, spk_y, spk_out  betas/thresholds (frozen)
TRAINED : D  — updated by unsupervised trace-based STDP

Key addition over the previous attempt
----------------------------------------
WTA (Winner-Takes-All) lateral inhibition in the spk_y population.
Without competition, all y-neurons tend to respond to every input and
the STDP updates cancel out in expectation, producing ~chance accuracy.
With WTA, only the first (strongest) y-neuron that crosses threshold
wins per timestep; the others are inhibited and cannot update D.
This forces neurons to specialise toward different input patterns,
exactly as in Diehl & Cook (2015).

Two-compartment adaptive threshold (homeostasis)
-------------------------------------------------
After a neuron wins, its personal threshold theta_adapt is raised by
`theta_plus`. It decays back toward zero between timesteps by factor
`theta_decay`. This prevents any single neuron from dominating
permanently, giving every neuron a fair chance to win — the biological
mechanism called intrinsic plasticity / homeostasis.

STDP rule (trace-based, additive)
-----------------------------------
For the synapse from input neuron i to y-neuron j:

  LTP: when spk_y[j] fires  →  dW += A_plus  * trace_pre[i]
  LTD: when spk_input[i] fires →  dW -= A_minus * trace_post[j]

Traces decay exponentially each timestep:
  trace_pre  *= tau_pre
  trace_post *= tau_post

Only the WINNING neuron (and its synapses) receives an update.
"""

import torch
import torch.nn as nn
import snntorch
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCellSTDPPhase1(LMUCore):
    """
    L2MU cell for Phase 1: unsupervised STDP on D, everything else frozen.

    Parameters (all passed through `params` dict)
    -----------------------------------------------
    memory_size       : number of y-neurons (= number of D rows)
    order             : LMU polynomial order
    theta             : LMU memory window
    beta_spk_m        : decay factor for spk_m neurons
    threshold_spk_m   : firing threshold for spk_m neurons
    beta_spk_y        : decay factor for spk_y neurons (base threshold)
    threshold_spk_y   : base firing threshold for spk_y (= test threshold)
    beta_spk_out      : decay factor for spk_out (unused in phase 1, kept for API compat.)
    threshold_spk_out : threshold for spk_out (unused in phase 1)
    stdp_a_plus       : LTP learning rate   (default 0.01)
    stdp_a_minus      : LTD learning rate   (default 0.01)
    stdp_tau_pre      : pre-synaptic trace decay  (default 0.95)
    stdp_tau_post     : post-synaptic trace decay (default 0.95)
    theta_plus        : homeostatic threshold increment per win (default 0.05)
    theta_decay       : homeostatic threshold decay per timestep (default 0.99)
    wta_inhibition    : inhibition strength applied to non-winners (default 1e9,
                        effectively hard inhibition)
    """

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

        # ── Freeze everything except D ──────────────────────────────────────
        # A, B, C are already non-trainable (requires_grad=False) from LMUCore.
        # We additionally freeze W_out and e_x so only D is updated by STDP.
        self.W_out.weight.requires_grad_(False)
        self.e_x.weight.requires_grad_(False)
        self.D.weight.requires_grad_(False)   # STDP handles D, not Adam

        # ── Spiking neuron populations ──────────────────────────────────────
        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"Neuron type '{neuron_type}' not found in snntorch.")

        # spk_m: memory population — frozen betas/thresholds in phase 1
        self.spk_m = Neuron(
            beta=params['beta_spk_m'],
            threshold=params['threshold_spk_m'],
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # spk_y: y-population — frozen betas/thresholds; WTA is applied manually
        self.spk_y = Neuron(
            beta=params['beta_spk_y'],
            threshold=params['threshold_spk_y'],
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # spk_out: kept for API compatibility but not used in phase 1
        self.spk_out = Neuron(
            beta=params.get('beta_spk_out', 0.9),
            threshold=params.get('threshold_spk_out', 1.0),
            learn_beta=False,
            learn_threshold=False,
            init_hidden=True,
        )

        # ── STDP hyperparameters ────────────────────────────────────────────
        self.stdp_a_plus   = float(params.get('stdp_a_plus',  0.01))
        self.stdp_a_minus  = float(params.get('stdp_a_minus', 0.01))
        self.stdp_tau_pre  = float(params.get('stdp_tau_pre',  0.95))
        self.stdp_tau_post = float(params.get('stdp_tau_post', 0.95))

        # Weight clipping range for D
        self.w_min = float(params.get('w_min', 0.0))
        self.w_max = float(params.get('w_max', 1.0))

        # ── Homeostatic threshold adaptation ────────────────────────────────
        # Each y-neuron has an adaptive threshold offset theta_adapt (on top of
        # the fixed base threshold stored in spk_y). When a neuron fires, its
        # theta_adapt increases by theta_plus, decaying back toward 0 over time.
        self.theta_plus  = float(params.get('theta_plus',  0.05))
        self.theta_decay = float(params.get('theta_decay', 0.99))
        # theta_adapt is per-neuron, shape [memory_size], initialised lazily
        self.theta_adapt = None   # will be [B, memory_size] on first use

        # ── WTA inhibition strength ──────────────────────────────────────────
        # Hard WTA: non-winning neurons get their membrane potential suppressed
        # by subtracting this large value before their threshold is checked.
        # 1e9 is effectively a hard reset for any reasonable threshold value.
        self.wta_inhibition = float(params.get('wta_inhibition', 1e9))

        # ── STDP traces (initialised lazily) ────────────────────────────────
        self.trace_pre  = None   # [B, input_size]
        self.trace_post = None   # [B, memory_size]

        # ── Spike history buffers (saved during forward for stdp_update) ─────
        self._spk_input_hist  = []   # list of [B, input_size]
        self._spk_y_wta_hist  = []   # list of [B, memory_size]  (post-WTA)

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def init_cell(self):
        """Reset all hidden states for a new sequence / batch."""
        self.spk_m.init_leaky()
        self.spk_y.init_leaky()
        self.spk_out.init_leaky()
        self.trace_pre  = None
        self.trace_post = None
        self.theta_adapt = None
        self._spk_input_hist = []
        self._spk_y_wta_hist = []
        return torch.empty(0)   # empty sentinel for spk_memory

    def _init_traces(self, batch_size, device):
        self.trace_pre  = torch.zeros(batch_size, self.input_size,  device=device)
        self.trace_post = torch.zeros(batch_size, self.memory_size, device=device)

    def _init_theta_adapt(self, batch_size, device):
        # Shape [B, memory_size] so we can handle per-sample homeostasis
        self.theta_adapt = torch.zeros(batch_size, self.memory_size, device=device)

    # ─────────────────────────────────────────────────────────────────────────
    # WTA lateral inhibition
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_wta(self, membrane_potential: torch.Tensor,
                   base_threshold: float) -> torch.Tensor:
        """
        Soft Winner-Takes-All with homeostatic threshold adaptation.

        Steps
        -----
        1. Compute effective threshold = base_threshold + theta_adapt
        2. Find which neurons exceed their effective threshold (candidates)
        3. Among candidates, pick the one with the highest membrane potential
           (the "winner"); inhibit all others by zeroing their output spike.
        4. If no neuron crosses threshold, no spike is emitted anywhere.
        5. Update theta_adapt: winner's threshold rises; all thresholds decay.

        Returns
        -------
        spk_wta : [B, memory_size]  binary spike tensor after WTA
        """
        # effective threshold per neuron: [B, memory_size]
        eff_thresh = base_threshold + self.theta_adapt   # broadcast over batch

        # candidates: neurons that cross their effective threshold
        candidates = (membrane_potential >= eff_thresh).float()  # [B, memory_size]

        # winner = candidate with max membrane potential (one per batch item)
        # If no candidate exists, max will still pick something — mask it out.
        masked_v = membrane_potential * candidates - 1e9 * (1.0 - candidates)
        winner_idx = masked_v.argmax(dim=1)  # [B]

        # Build one-hot winner mask
        spk_wta = torch.zeros_like(membrane_potential)
        spk_wta.scatter_(1, winner_idx.unsqueeze(1), 1.0)
        # Zero out winners that had NO candidate at all (no neuron fired)
        any_fired = candidates.sum(dim=1, keepdim=True) > 0   # [B, 1]
        spk_wta = spk_wta * any_fired.float()

        # ── Homeostatic threshold update ────────────────────────────────────
        # Winners: theta_adapt += theta_plus
        # All neurons: theta_adapt *= theta_decay  (applied after increment)
        self.theta_adapt = (self.theta_adapt + self.theta_plus * spk_wta) * self.theta_decay

        return spk_wta

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, spk_input: torch.Tensor,
                spk_memory: torch.Tensor):
        """
        Forward pass — no weight modifications here.
        Spike history is saved for stdp_update_D() which runs after each batch.

        spk_input  : [B, input_size]   — spiking input at timestep t
        spk_memory : [B, memory_size, order]  — LMU state

        Returns
        -------
        spk_out    : [B, output_size]  — kept for API compatibility (not used)
        spk_memory : [B, memory_size, order]  — updated LMU state
        """
        batch_size = spk_input.shape[0]
        device     = spk_input.device

        if spk_memory.numel() == 0:
            spk_memory = torch.zeros(
                (batch_size, self.memory_size, self.order),
                dtype=torch.float, device=device)

        if self.trace_pre is None:
            self._init_traces(batch_size, device)

        if self.theta_adapt is None:
            self._init_theta_adapt(batch_size, device)

        # ── m population (LMU memory update) ────────────────────────────────
        u_t    = self.e_x(spk_input)          # [B, memory_size]
        u_t_3d = u_t.unsqueeze(-1)            # [B, memory_size, 1]
        curr_m = self.A(spk_memory) + self.B(u_t_3d)
        spk_memory = self.spk_m(curr_m)       # [B, memory_size, order]

        # ── y population (raw membrane potential, then WTA) ──────────────────
        curr_y = self.C(spk_memory).squeeze(-1) + self.D(spk_input)
        # Pass through snntorch neuron to get membrane potential;
        # we grab the membrane potential directly for WTA.
        # snntorch.Leaky stores mem in ._mem after a forward call.
        spk_y_raw = self.spk_y(curr_y)        # [B, memory_size]  (pre-WTA output)

        # Retrieve the internal membrane potential from snntorch
        # (spk_y._mem is updated after spk_y() is called)
        mem_y = self.spk_y.mem.detach()       # [B, memory_size]

        # Apply WTA + homeostasis
        base_thresh = float(self.spk_y.threshold)
        spk_y_wta = self._apply_wta(mem_y, base_thresh)   # [B, memory_size]

        # ── out population (API compatibility only) ──────────────────────────
        curr_out = self.W_out(spk_y_wta)
        spk_out  = self.spk_out(curr_out)

        # ── Save spike history for STDP (no gradient needed) ─────────────────
        if self.training:
            self._spk_input_hist.append(spk_input.detach())
            self._spk_y_wta_hist.append(spk_y_wta.detach())

        return spk_out, spk_memory

    # ─────────────────────────────────────────────────────────────────────────
    # STDP update for D (called AFTER the full sequence / batch)
    # ─────────────────────────────────────────────────────────────────────────

    def stdp_update_D(self):
        """
        Apply unsupervised trace-based STDP to D using the spike history
        accumulated during the last forward pass(es).

        Must be called ONCE per batch, after the full temporal sequence has
        been processed. No loss.backward() is needed or used here.

        STDP rule (additive, per-winner):
            LTP: winner j fired after pre-neuron i was active
                 →  D[j, i] += A_plus * trace_pre[i]
            LTD: pre-neuron i fires while winner j was recently active
                 →  D[j, i] -= A_minus * trace_post[j]

        Only the winning neuron at each timestep participates in the update.
        """
        if not self._spk_input_hist:
            return   # validation/test — nothing to do

        with torch.no_grad():
            for spk_input, spk_y_wta in zip(
                self._spk_input_hist,
                self._spk_y_wta_hist,
            ):
                # ── Trace decay ──────────────────────────────────────────────
                self.trace_pre  = self.stdp_tau_pre  * self.trace_pre  + spk_input
                self.trace_post = self.stdp_tau_post * self.trace_post + spk_y_wta

                # ── LTP: winner fired → strengthen connections that were active ─
                # dW_plus[j, i] = A_plus * spk_y_wta[b,j] * trace_pre[b,i]
                # averaged over batch
                dW_plus = self.stdp_a_plus * torch.einsum(
                    'bj,bi->ji', spk_y_wta, self.trace_pre
                ) / spk_input.shape[0]

                # ── LTD: pre fired → weaken connections to recently active winners
                # dW_minus[j, i] = A_minus * trace_post[b,j] * spk_input[b,i]
                dW_minus = self.stdp_a_minus * torch.einsum(
                    'bj,bi->ji', self.trace_post, spk_input
                ) / spk_input.shape[0]

                self.D.weight.add_(dW_plus - dW_minus)

            # ── Weight clipping to [w_min, w_max] ────────────────────────────
            self.D.weight.clamp_(self.w_min, self.w_max)

        # Clear history for next batch
        self._spk_input_hist = []
        self._spk_y_wta_hist = []
