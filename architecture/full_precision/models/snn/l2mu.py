import torch
import torch.nn as nn
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell


class L2MU(nn.Module):
    def __init__(self, input_size, output_size, params, neuron_type='Leaky'):
        super().__init__()
        self.l2mu_cell = L2MUCell(
            input_size=input_size,
            output_size=output_size,
            params=params,
            neuron_type=neuron_type,
        )

    def forward(self, input):
        """
        Args:
            input: [T, B, input_size]  (time-first, batch second)
        Returns:
            stacked spk_out: [T, B, output_size]
        """
        spk_memory = self.l2mu_cell.init_cell()   # empty tensor → zeros on first step

        spk_out_list = []
        for step in range(input.size(0)):
            spk_out, spk_memory = self.l2mu_cell(
                input[step].flatten(1),   # [B, input_size]
                spk_memory=spk_memory,
            )
            spk_out_list.append(spk_out)

        return torch.stack(spk_out_list, dim=0)  # [T, B, output_size]