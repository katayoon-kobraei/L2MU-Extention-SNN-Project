"""
Experiment 1 — Variant A + WTA (new run).

Re-runs Exp1 with two key changes from the previous run:
    1. W_in is a fixed random projection — never STDP'd.
       (Reproduces Variant A which gave D std=0.25 vs std=0.06
        for the simultaneous W_in+D run currently saved.)
    2. WTA applied during STDP: only the winning y neuron's D
       weights update per timestep, forcing each neuron to
       specialize toward a different Braille pattern.

Expected result:
    D mean ≈ 0.28, D std ≈ 0.25, 25–30 active neurons out of 250
    (Variant A) — or better with WTA pushing further specialization.

Weights saved to: model_insights/results/exp1_wta/
    D_weights.pt    → use in Exp2B, Exp2C, Exp2D (update their paths)
    W_in_weights.pt → use in Exp2C, Exp2D (update their paths)
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


def train_exp1_wta(
    params,
    num_epochs=150,
    data_dir="data/braille_full_splitted",
    split=2,
    save_dir="model_insights/results/exp1_wta",
):
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    data_module = BRAILLE(
        data_dir=data_dir,
        batch_size=int(params['batch_size']),
        split=split,
    )
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    print(f"Training samples: {len(data_module.train_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = L2MU(
        input_size=data_module.num_inputs,
        params=params,
        neuron_type='Leaky',
    ).to(device)

    print(f"\nD shape   : {model.l2mu_cell.D.weight.shape}")
    print(f"W_in shape: {model.l2mu_cell.W_in.weight.shape}")
    print(f"\nInitial D   — mean: {model.l2mu_cell.D.weight.mean():.4f}  "
          f"std: {model.l2mu_cell.D.weight.std():.4f}")
    print(f"Initial W_in — mean: {model.l2mu_cell.W_in.weight.mean():.4f}  "
          f"std: {model.l2mu_cell.W_in.weight.std():.4f}  (fixed, never updated)")
    print(f"\nthreshold_spk_y : {params['threshold_spk_y']}")
    print(f"WTA             : enabled (curr_y argmax)")
    print(f"W_in STDP       : same strategy")

    # ── Training: D only, 150 epochs ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training D with unsupervised STDP + WTA (W_in frozen)")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):

        total_y_rate     = 0.0
        total_wta_rate   = 0.0   # rate after WTA mask (should be ≤ 1 active/sample)
        num_batches      = 0

        for data, _ in train_loader:
            data = data.to(device).swapaxes(0, 1)   # [T, B, 24]

            with torch.no_grad():
                spk_y_stack = model(data)            # [T, B, 250] raw spk_y

            # spk_y_wta rate: average active neurons per sample per step
            # Derived from history saved inside the cell
            total_y_rate  += spk_y_stack.mean().item()
            num_batches   += 1

            model.l2mu_cell.stdp_update()

        avg_y_rate = total_y_rate / num_batches
        D_mean     = model.l2mu_cell.D.weight.mean().item()
        D_std      = model.l2mu_cell.D.weight.std().item()
        D_min      = model.l2mu_cell.D.weight.min().item()
        D_max      = model.l2mu_cell.D.weight.max().item()

        # Count neurons with std > 0.05 (rough proxy for "active / specialized")
        # A neuron is "specialized" if its 24 input weights are non-uniform
        D_per_neuron_std = model.l2mu_cell.D.weight.std(dim=1)   # [250]
        active_neurons   = (D_per_neuron_std > 0.05).sum().item()

        Win_mean = model.l2mu_cell.W_in.weight.mean().item()
        Win_std  = model.l2mu_cell.W_in.weight.std().item()
        print(
            f"Ep {epoch+1:3d}/{num_epochs} | "
            f"y rate: {avg_y_rate:.4f} | "
            f"D mean: {D_mean:.4f}  std: {D_std:.4f} | "
            f"W_in mean: {Win_mean:.4f}  std: {Win_std:.4f} | "
            f"active neurons: {active_neurons}/250"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.l2mu_cell.D.weight.data,   save_path / 'D_weights.pt')
    torch.save(model.l2mu_cell.W_in.weight.data, save_path / 'W_in_weights.pt')
    torch.save(model.state_dict(),               save_path / 'model_exp1_wta.pt')

    print(f"\nD weights saved to   : {save_path / 'D_weights.pt'}")
    print(f"W_in weights saved to: {save_path / 'W_in_weights.pt'}")
    print(f"\nFinal D — mean: {model.l2mu_cell.D.weight.mean():.4f}  "
          f"std: {model.l2mu_cell.D.weight.std():.4f}")
    print(f"\nNext step: update D_weights_path and W_in_weights_path in")
    print(f"  train_exp2b.py, train_exp2c.py, train_exp2d.py")
    print(f"  to point to: {save_dir}/")

    return model


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'order': 4.0,
        'theta': 7.0,
        'memory_size': 250.0,
        # m population
        'beta_spk_m': 0.35,
        'threshold_spk_m': 0.4,
        # y population — threshold=2.5 matches Variant A (rate ~0.138)
        # Lower threshold → more neurons fire → more WTA competition per step
        'beta_spk_y': 0.3,
        'threshold_spk_y': 7.0,
    }

    train_exp1_wta(
        params=params,
        num_epochs=150,
        data_dir="data/braille_full_splitted",
        split=2,
        save_dir="model_insights/results/exp1_wta",
    )