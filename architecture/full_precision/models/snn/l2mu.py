import torch
import torch.nn as nn
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell


class L2MU(nn.Module):
    """
    Experiment 2B model — full SSM with S2-STDP supervised output layer.

    Populations:
        m + y  : D frozen from Exp1
        out    : 54 LIF neurons (2 per Braille class), W_out trained by S2-STDP
                 Neuron 2*c   = target neuron for class c
                 Neuron 2*c+1 = non-target neuron for class c

    Classification at inference:
        For each class c, the "vote" is the spike count of its target neuron
        (index 2*c).  The predicted class is argmax over target neurons.
        (Non-target neurons are used during training only to improve class
        separation and are not used for the final prediction.)
    """

    def __init__(self, input_size, params, neuron_type='Leaky'):
        super().__init__()
        self.l2mu_cell = L2MUCell(
            input_size=input_size,
            params=params,
            neuron_type=neuron_type,
        )
        self.num_classes = L2MUCell.NUM_CLASSES

    def forward(self, input_seq: torch.Tensor):
        """
        Full T-step forward pass over a spike sequence.

        Args:
            input_seq : [T, B, input_size]

        Returns:
            spk_out_stack : [T, B, 54]  — output spikes at each timestep
            spk_y_stack   : [T, B, memory_size]
        """
        spk_memory = self.l2mu_cell.init_cell()

        spk_out_list = []
        spk_y_list   = []

        for step in range(input_seq.size(0)):
            spk_out, spk_y, spk_memory = self.l2mu_cell(
                input_seq[step].flatten(1),
                spk_memory=spk_memory,
            )
            spk_out_list.append(spk_out)
            spk_y_list.append(spk_y)

        return (
            torch.stack(spk_out_list, dim=0),   # [T, B, 54]
            torch.stack(spk_y_list,   dim=0),   # [T, B, memory_size]
        )

    def predict(self, spk_out_stack: torch.Tensor) -> torch.Tensor:
        """
        Convert output spike stack to class predictions.

        Strategy: for each class c, use the total spike count of target
        neuron 2*c over all timesteps.  Predicted class = argmax.

        Args:
            spk_out_stack : [T, B, 54]

        Returns:
            predictions : [B]  — integer class indices
        """
        # Sum spikes over time: [B, 54]
        spike_counts = spk_out_stack.sum(dim=0)

        # Extract only target neurons (indices 0, 2, 4, ..., 52)
        target_indices = torch.arange(
            0, L2MUCell.NUM_OUT, L2MUCell.NEURONS_PER_CLS,
            device=spike_counts.device
        )   # [27]
        target_counts = spike_counts[:, target_indices]   # [B, 27]

        return target_counts.argmax(dim=1)   # [B]