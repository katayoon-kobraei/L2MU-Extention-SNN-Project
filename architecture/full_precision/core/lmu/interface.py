import torch
from torch import nn
import numpy as np
from abc import abstractmethod
from architecture.full_precision.core.lmu.utils import LCLinear, CLinear, XavierLinear

class LMUCore(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            memory_size,
            order,
            theta,
            output_size=None,
            output=False,
            bias=False,
            trainable_theta=False,
            discretizer='zoh'

    ):
        super().__init__()

        self.B = None
        self.A = None
        self.C = None
        self.D = None

        # Parameters passed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.order = order
        self._init_theta = theta
        self.output_size = output_size
        self.bias = bias
        self.output = output
        self.trainable_theta = trainable_theta
        self.discretizer = discretizer

        # Parameters to be learned
        self.W_h = None
        self.W_m = None
        self.W_x = None
        self.bias_m = None
        self.bias_h = None
        self.bias_x = None
        self.e_m = None
        self.e_h = None
        self.e_x = None
        self.theta_inv = None
        self.output_transformation = None

    def init_parameters(self):
        if self.trainable_theta:
            self.theta_inv = nn.Parameter(torch.empty(()))
        else:
            self.theta_inv = 1 / self._init_theta

        self.e_x = LCLinear(self.input_size, self.memory_size, bias=False)
        self.e_h = LCLinear(self.hidden_size, self.memory_size, bias=False)
        self.e_m = CLinear(self.memory_size * self.order, self.memory_size, bias=False)
        # Kernels

        self.A = CLinear(self.order, self.order, bias=False)
        self.B = CLinear(1, self.order, bias=False)
        self._gen_AB()

        self.W_x = XavierLinear(self.input_size, self.hidden_size, bias=False)
        self.W_h = XavierLinear(self.hidden_size, self.hidden_size, bias=False)
        self.W_m = XavierLinear(self.memory_size * self.order, self.hidden_size, bias=False)

        if self.output: # spk_memory:[batch, memory_size * order]   spk_u:[batch, memory_size] 
            self.output_transformation = nn.Linear(self.memory_size, self.output_size)
            self.C = CLinear(self.order, 1, bias=False)
            self.D = CLinear(1, 1, bias=False)
            self._gen_CD()
    @property
    def theta(self):
        if self.trainable_theta:
            return 1 / self.theta_inv
        return self._init_theta

    def _gen_AB(self):
        """Generates A and B matrices."""

        # compute analog A/B matrices
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # discretize matrices
        if self.discretizer == "zoh":
            # save the un-discretized matrices for use in .call
            _base_A = torch.FloatTensor(A.T)
            _base_B = torch.FloatTensor(B.T)

            if self.trainable_theta:
                self._base_A = nn.Parameter(_base_A, requires_grad=False)
                self._base_B = nn.Parameter(_base_B, requires_grad=False)
            else:
                self._base_A = torch.tensor(0)
                self._base_B = torch.tensor(0)

            A, B = self._cont2discrete_zoh(
                _base_A / self._init_theta, _base_B / self._init_theta
            )

            self.A.weight = nn.Parameter(A, requires_grad=False)
            self.B.weight = nn.Parameter(B, requires_grad=False)

        else:
            if not self.trainable_theta:
                A = A.T / self._init_theta + np.eye(self.order)
                B = B.T / self._init_theta

            self.A.weight = nn.Parameter(A, requires_grad=False)
            self.B.weight = nn.Parameter(B, requires_grad=False)


    def _shifted_legendre(self, n, x):
        """Shifted Legendre polynomial P_n*(x), x in [0,1]."""
        if n == 0:
            return 1.0
        if n == 1:
            return 2.0 * x - 1.0

        p_nm1 = 1.0
        p_n = 2.0 * x - 1.0
        for k in range(1, n):
            p_np1 = ((2 * k + 1) * (2 * x - 1) * p_n - k * p_nm1) / (k + 1)
            p_nm1, p_n = p_n, p_np1
        return p_n


    def _gen_CD(self, theta_prime=None):
        """
        Generates fixed C and D matrices for the SSM output:
            y[t] = C x[t] + D u[t]

        theta_prime: delay to reconstruct inside the memory window.
                    If None, defaults to full delay theta.
        """
        if theta_prime is None:
            theta_prime = self._init_theta

        alpha = float(theta_prime) / float(self._init_theta)  # in [0,1]

        # C has one row: [P0(alpha), P1(alpha), ..., P_{order-1}(alpha)]
        C = np.array(
            [[self._shifted_legendre(i, alpha) for i in range(self.order)]],
            dtype=np.float32
        )  # shape (1, order)

        # For the LDN formulation, D = 0
        D = np.zeros((1, 1), dtype=np.float32)

        self.C.weight = nn.Parameter(torch.tensor(C), requires_grad=False)
        self.D.weight = nn.Parameter(torch.tensor(D), requires_grad=False)


    @staticmethod
    def _cont2discrete_zoh(A, B):
        """
        Function to discretize A and B matrices using Zero Order Hold method.

        Functionally equivalent to
        ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``
        (but implemented in Pytorch so that it is differentiable).

        Note that this accepts and returns matrices that are transposed from the
        standard linear system implementation (as that makes it easier to use in
        `.call`).
        """

        # combine A/B and pad to make square matrix
        em_upper = torch.concat([A, B], dim=0)  # pylint: disable=no-value-for-parameter
        padding = (0, B.shape[0], 0, 0)
        em = torch.nn.functional.pad(em_upper, padding)

        # compute matrix exponential
        ms = torch.matrix_exp(em)

        # slice A/B back out of combined matrix
        discreet_A = ms[: A.shape[0], : A.shape[1]]
        discreet_B = ms[A.shape[0]:, : A.shape[1]]
        discreet_B = discreet_B.reshape(discreet_B.shape[1], discreet_B.shape[0])

        return discreet_A, discreet_B

    @classmethod
    @abstractmethod
    def forward(self, input_, _h, _m):
        pass
