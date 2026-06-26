"""
Experiment 1: Unsupervised STDP training of D matrix only.

No output layer. No labels. No cross-entropy loss.
D learns internal representations of Braille letters
purely from spike timing correlations.

Speed fix: process full batches together instead of sample by sample.
Traces reset at the start of each batch (between batches not samples).
This is fast and still produces meaningful STDP learning.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from pathlib import Path
from lightning.pytorch import seed_everything
from data.BRAILLE import BRAILLE
from architecture.full_precision.models.snn.l2mu import L2MU

torch.set_float32_matmul_precision('high')


def train_exp1(params, PHASE1_EPOCHS=75, PHASE2_EPOCHS=75, data_dir="../data/braille_full_splitted",
               split=2, save_dir="../model_insights/results/exp1"):

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data ---
    data_module = BRAILLE(
        data_dir=data_dir,
        batch_size=int(params['batch_size']),
        split=split,
    )
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Input size: {data_module.num_inputs}")

    # --- Model ---
    model = L2MU(
        input_size=data_module.num_inputs,
        params=params,
        neuron_type='Leaky',
    ).to(device)

    print(f"\nModel D weight shape: {model.l2mu_cell.D.weight.shape}")
    print(f"Initial D weight mean: {model.l2mu_cell.D.weight.mean().item():.4f}")
    print(f"Initial D weight std:  {model.l2mu_cell.D.weight.std().item():.4f}")

    print(f"Model W_in weight shape: {model.l2mu_cell.W_in.weight.shape}")
    print(f"Initial W_in weight mean: {model.l2mu_cell.W_in.weight.mean().item():.4f}")
    print(f"Initial W_in weight std:  {model.l2mu_cell.W_in.weight.std().item():.4f}")


    model.l2mu_cell._train_Win = True
    model.l2mu_cell._train_D   = True

    for epoch in range(150):
        total_spike_rate = 0.0
        num_batches = 0

        for data, _ in train_loader:
            data = data.to(device).swapaxes(0, 1)
            with torch.no_grad():
                spk_y = model(data)
            total_spike_rate += spk_y.mean().item()
            num_batches += 1
            model.l2mu_cell.stdp_update()

        D_mean   = model.l2mu_cell.D.weight.mean().item()
        D_std    = model.l2mu_cell.D.weight.std().item()
        W_in_mean = model.l2mu_cell.W_in.weight.mean().item()
        W_in_std  = model.l2mu_cell.W_in.weight.std().item()
        avg_rate = total_spike_rate / num_batches

        print(
            f"Epoch {epoch+1:3d}/150 | "
            f"y rate: {avg_rate:.4f} | "
            f"D mean: {D_mean:.4f} std: {D_std:.4f} | "
            f"W_in mean: {W_in_mean:.4f} std: {W_in_std:.4f}"
        )

        

    # --- Save D weights ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.l2mu_cell.D.weight.data, save_path / 'D_weights.pt')
    torch.save(model.l2mu_cell.W_in.weight.data, save_path / 'W_in_weights.pt')
    torch.save(model.state_dict(), save_path / 'model_exp1.pt')
    print(f"\nD weights saved to: {save_path / 'D_weights.pt'}")

    return model


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'order': 4.0,
        'theta': 7.0,
        'memory_size': 250.0,
        'beta_spk_m': 0.35,
        'threshold_spk_m': 0.4,
        'beta_spk_y': 0.3,
        'threshold_spk_y': 2.5,  # higher threshold to control spike rate
    }

    train_exp1(
        params=params,
        data_dir="data/braille_full_splitted",
        split=2,
        save_dir="model_insights/results/exp1",
    )
