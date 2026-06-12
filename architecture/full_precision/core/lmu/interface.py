import torch
from torch import nn
import numpy as np
from abc import abstractmethod
from architecture.full_precision.core.lmu.utils import LCLinear, CLinear, XavierLinear


class LMUCore(nn.Module):
    """
    Pure SSM core implementing:
        m[t+1] = A * m_spk[t] + B * x_spk[t]   (m population)
        y[t]   = C * m_spk[t] + D * x_spk[t]   (y population)

    A, B, C are fixed matrices derived from the LMU/LDN formulation.
    D is a generic trainable weight matrix.

    Removed from the original L2MU:
        - e_x, e_h, e_m  (encoding vectors)
        - W_x, W_h, W_m  (hidden-state kernels)
        - hidden_size / hidden state entirely
        - output_transformation linear layer
    """

    def __init__(
            self,
            input_size,
            memory_size,
            order,
            theta,
            output_size,
            trainable_theta=False,
            discretizer='zoh',
    ):
        super().__init__()

        # Fixed SSM matrices
        self.A = None
        self.B = None
        self.C = None
        self.D = None   # trainable
        self.W_out = None  # trainable

        # Dimensions
        self.input_size      = input_size
        self.memory_size     = memory_size
        self.order           = order
        self._init_theta     = theta
        self.output_size     = output_size
        self.trainable_theta = trainable_theta
        self.discretizer     = discretizer

    def init_parameters(self):

        if self.trainable_theta:
            self.theta_inv = nn.Parameter(torch.empty(()))
        else:
            self.theta_inv = 1.0 / self._init_theta

        # A and B: fixed, LMU-derived
        self.A = CLinear(self.order, self.order, bias=False)
        self.B = CLinear(1, self.order, bias=False)
        self._gen_AB()

        # C: fixed, LDN Legendre projection [1, order]
        self.C = CLinear(self.order, 1, bias=False)
        self._gen_C()

        # D: trainable [memory_size, memory_size]
        # D now receives u_t = W_in(spk_input) instead of raw spk_input,
        # so its input dimension changes from input_size (24) to memory_size (250).
        self.D = XavierLinear(self.input_size, self.memory_size, bias=False)


        # W_out: trainable [output_size, memory_size]
        self.W_out = XavierLinear(self.memory_size, self.output_size, bias=False)

        # W_in: input encoder [input_size -> memory_size]
        # Projects spiking input to one scalar per memory slot (u(t) in LMU)
        self.W_in = XavierLinear(self.input_size, self.memory_size, bias=False)

    @property
    def theta(self):
        if self.trainable_theta:
            return 1 / self.theta_inv
        return self._init_theta

    def _gen_AB(self):
        """Generates fixed A and B matrices from the LMU formulation."""
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        if self.discretizer == 'zoh':
            _base_A = torch.FloatTensor(A.T)
            _base_B = torch.FloatTensor(B.T)

            if self.trainable_theta:
                self._base_A = nn.Parameter(_base_A, requires_grad=False)
                self._base_B = nn.Parameter(_base_B, requires_grad=False)
            else:
                self._base_A = torch.tensor(0)
                self._base_B = torch.tensor(0)

            A_disc, B_disc = self._cont2discrete_zoh(
                _base_A / self._init_theta,
                _base_B / self._init_theta,
            )
            self.A.weight = nn.Parameter(A_disc, requires_grad=False)
            self.B.weight = nn.Parameter(B_disc, requires_grad=False)

        else:   # Euler
            if not self.trainable_theta:
                A = A.T / self._init_theta + np.eye(self.order)
                B = B.T / self._init_theta
            self.A.weight = nn.Parameter(torch.FloatTensor(A), requires_grad=False)
            self.B.weight = nn.Parameter(torch.FloatTensor(B), requires_grad=False)

    def _shifted_legendre(self, n, x):
        """
        Evaluates the n-th shifted Legendre polynomial P_n*(x) at x in [0,1].

        Recurrence:
            P_0*(x) = 1
            P_1*(x) = 2x - 1
            P_{k+1}*(x) = ((2k+1)(2x-1) P_k*(x) - k P_{k-1}*(x)) / (k+1)

        The row vector [P_0*(alpha), ..., P_{d-1}*(alpha)] is the C matrix
        that reconstructs u(t - theta') from m(t), where alpha = theta'/theta.
        This matches Eq. 25 of the DeepLSNN supplementary (Eq. 3 of LMU paper).
        """
        if n == 0:
            return 1.0
        if n == 1:
            return 2.0 * x - 1.0
        p_prev, p_curr = 1.0, 2.0 * x - 1.0
        for k in range(1, n):
            p_next = ((2 * k + 1) * (2 * x - 1) * p_curr - k * p_prev) / (k + 1)
            p_prev, p_curr = p_curr, p_next
        return p_curr

    def _gen_C(self, theta_prime=None):
        """
        Builds the fixed C matrix as a row of shifted Legendre polynomial values.

            C shape: [1, order]
            C[0, i] = P_i*(alpha),  alpha = theta_prime / theta

        alpha = 1  (default, full delay) means reconstructing the input at
        the oldest point in the memory window — the standard LDN readout.
        """
        if theta_prime is None:
            theta_prime = self._init_theta   # full delay → alpha = 1

        alpha = float(theta_prime) / float(self._init_theta)   # in [0, 1]

        C = np.array(
            [[self._shifted_legendre(i, alpha) for i in range(self.order)]],
            dtype=np.float32,
        )   # shape [1, order]

        self.C.weight = nn.Parameter(torch.tensor(C), requires_grad=False)

    @staticmethod
    def _cont2discrete_zoh(A, B):
        """Discretise A and B using Zero-Order Hold (ZOH). Unchanged from original."""
        em_upper = torch.concat([A, B], dim=0)
        padding  = (0, B.shape[0], 0, 0)
        em       = torch.nn.functional.pad(em_upper, padding)
        ms       = torch.matrix_exp(em)
        disc_A   = ms[: A.shape[0], : A.shape[1]]
        disc_B   = ms[A.shape[0]:,  : A.shape[1]]
        disc_B   = disc_B.reshape(disc_B.shape[1], disc_B.shape[0])
        return disc_A, disc_B

    @abstractmethod
    def forward(self, spk_input, spk_memory):
        pass
