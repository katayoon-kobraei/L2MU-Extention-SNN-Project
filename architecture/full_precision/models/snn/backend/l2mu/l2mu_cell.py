import snntorch
import torch
from architecture.full_precision.core.lmu.interface import LMUCore


class L2MUCell(LMUCore):

    def __init__(
            self,
            input_size,
            params,
            bias=False,
            trainable_theta=False,
            output_size=None,
            output=False,
            neuron_type='Leaky',
    ):

        super().__init__(input_size=input_size, hidden_size=int(params['hidden_size']),
                         memory_size=int(params['memory_size']), order=int(params['order']), theta=params['theta'],
                         output=output, output_size=output_size, bias=bias, trainable_theta=trainable_theta)

        # self._gen_AB()
        self.init_parameters()

        try:
            Neuron = getattr(snntorch, neuron_type)
        except AttributeError:
            raise ValueError(f"The neuron type '{neuron_type}' does not exist in snntorch. Check the class name.")

##### Flag: init_hidden = True?
        self.spk_u = Neuron(beta=params['beta_spk_u'], threshold=params['threshold_spk_u'], learn_beta=True, learn_threshold=True, init_hidden=True,)
        self.spk_h = Neuron(beta=params['beta_spk_h'], threshold=params['threshold_spk_h'],  learn_beta=True, learn_threshold=True, init_hidden=True)
        self.spk_m = Neuron(beta=params['beta_spk_m'], threshold=params['threshold_spk_m'], learn_beta=True, learn_threshold=True, init_hidden=True)

        if self.output:
            self.spk_output = Neuron(beta=params['beta_spk_output'], threshold=params['threshold_spk_output'],  learn_beta=True, learn_threshold=True, init_hidden=True)


    def init_cell(self):
        self.spk_u.init_leaky()
        self.spk_h.init_leaky()
        self.spk_m.init_leaky()

        if self.output:
            self.spk_output.init_leaky()

        spike_h = torch.empty(0)
        spike_m = torch.empty(0)

        return spike_h, spike_m

    def forward(self, spk_input: torch.Tensor , spk_hidden: torch.Tensor, spk_memory: torch.Tensor):

        if spk_hidden.numel() == 0 or spk_memory.numel() == 0:
            batch_size = spk_input.shape[0]  # Get the batch size from spk_input
            spk_hidden = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, device=spk_input.device)
            spk_memory = torch.zeros((batch_size, self.memory_size * self.order), dtype=torch.float, device=spk_input.device)

            # Enable gradients for both tensors
            spk_hidden.requires_grad_(True)
            spk_memory.requires_grad_(True)

        # Equation (7) of the paper
        curr_u = self.e_x(spk_input) + self.e_h(spk_hidden) + self.e_m(spk_memory)

        # Equation (4) of the paper
        spk_u = self.spk_u(curr_u)
        spk_u_flat = spk_u

        # separate memory/order dimensions
        spk_u = torch.unsqueeze(spk_u, -1)   # [B, memory_size, 1]
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size, self.order))

        curr_m = self.A(spk_memory) +  self.B(spk_u)

        if self.discretizer == 'euler' and self.trainable_theta:
            curr_m += curr_m * self.theta_inv

        spk_memory = self.spk_m(curr_m)

            # Hidden-state path (kept unchanged if you still want L2MU internal dynamics)
        spk_memory_flat = torch.reshape(
            spk_memory, (-1, self.memory_size * self.order)
        )                                              # [B, memory_size * order]

        curr_h = self.W_x(spk_input) + self.W_h(spk_hidden) + self.W_m(spk_memory_flat)
        spk_hidden = self.spk_h(curr_h)                # [B, hidden_size]

        # Core SSM output: y[t] = C x[t] + D u[t]
        if self.output:
            y_ssm = self.C(spk_memory) + self.D(spk_u)   # [B, memory_size, 1]
            y_ssm = y_ssm.squeeze(-1)                    # [B, memory_size]

            curr_output = self.output_transformation(y_ssm)

            # Better to return logits directly for classification
            return curr_output, spk_hidden, spk_memory_flat

        return spk_hidden, spk_hidden, spk_memory_flat
