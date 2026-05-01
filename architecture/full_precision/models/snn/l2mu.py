import torch
import torch.nn as nn
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell


class L2MU(nn.Module):
    """
    Experiment 2A model — full SSM with label assignment output.
    m + y populations (D frozen from Exp1) + out population (27 neurons, STDP).
    """

    def __init__(self, input_size, params, neuron_type='Leaky'):
        super().__init__()
        self.l2mu_cell = L2MUCell(
            input_size=input_size,
            params=params,
            neuron_type=neuron_type,
        )

    def forward(self, input):
        """
        Args:
            input: [T, B, input_size]
        Returns:
            spk_out_stack: [T, B, 27]
            spk_y_stack:   [T, B, memory_size]
        """
        spk_memory = self.l2mu_cell.init_cell()

        spk_out_list = []
        spk_y_list   = []

        for step in range(input.size(0)):
            spk_out, spk_y, spk_memory = self.l2mu_cell(
                input[step].flatten(1),
                spk_memory=spk_memory,
            )
            spk_out_list.append(spk_out)
            spk_y_list.append(spk_y)

        return (
            torch.stack(spk_out_list, dim=0),   # [T, B, 27]
            torch.stack(spk_y_list,   dim=0),   # [T, B, memory_size]
        )
