"""
Experiment 2B: Supervised S2-STDP output layer — rate-code adaptation.

Architecture:
    - m + y populations: frozen (D loaded from Exp1)
    - out population: 54 LIF neurons (2 per Braille class)
        * Neuron 2*c   = target neuron for class c     → learns to fire MORE
        * Neuron 2*c+1 = non-target neuron for class c → learns to fire LESS
    - W_out trained with rate-code S2-STDP (supervised, labels used per batch)

S2-STDP rule (rate-code adaptation of Goupy et al. Frontiers 2024):
    rate_j   = count_j / T  ∈ [0, 1]
    R_mean   = mean(rate_j) over all 54 output neurons (per sample)

    Desired rates:
        target     → R_mean + (C-1)/C * g   (fire more)
        non-target → R_mean - 1/C * g       (fire less)

    Error:
        e_j = rate_j - rate_desired_j

    Weight update (Hebbian, error-modulated):
        dW[j,i] = -e_j * lr * rate_pre_i
                        (potentiate if firing too little, depress if too much)

Classification at inference:
    Predict = argmax over spike counts of target neurons (2*c for each class c).
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
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell

torch.set_float32_matmul_precision('high')

NUM_CLASSES = 27


def train_exp2b(
    params,
    num_epochs=150,
    data_dir="data/braille_full_splitted",
    split=2,
    D_weights_path="model_insights/results/exp1/D_weights.pt",
    save_dir="model_insights/results/exp2b",
):
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    data_module = BRAILLE(
        data_dir=data_dir,
        batch_size=int(params['batch_size']),
        split=split,
    )
    data_module.setup('fit')
    data_module.setup('test')
    train_loader = data_module.train_dataloader()
    test_loader  = data_module.test_dataloader()
    print(f"Training samples : {len(data_module.train_dataset)}")
    print(f"Test samples     : {len(data_module.test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = L2MU(
        input_size=data_module.num_inputs,
        params=params,
        neuron_type='Leaky',
    ).to(device)

    model.l2mu_cell.load_D_from_exp1(D_weights_path)

    print(f"\nW_out shape     : {model.l2mu_cell.W_out.weight.shape}")
    print(f"Output neurons  : {L2MUCell.NUM_OUT} "
          f"({L2MUCell.NEURONS_PER_CLS} per class × {NUM_CLASSES} classes)")
    print(f"threshold_spk_y  : {params['threshold_spk_y']}")
    print(f"threshold_spk_out: {params['threshold_spk_out']}")
    print(f"S2STDP_G         : {L2MUCell.S2STDP_G}")
    print(f"S2STDP_A_LR      : {L2MUCell.S2STDP_A_LR}")

    # ── Sanity check: y-population spike rate before training ─────────────
    print("\n── Sanity check: y-population spike rate (first batch) ──")
    model.eval()
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device).swapaxes(0, 1)
            spk_out, spk_y = model(data)
            T = data.shape[0]
            y_rate   = spk_y.sum(dim=0).mean().item() / T
            out_rate = spk_out.sum(dim=0).mean().item() / T
            tgt_rate = spk_out.sum(dim=0)[:, ::2].mean().item() / T
            nontgt_rate = spk_out.sum(dim=0)[:, 1::2].mean().item() / T
            print(f"  y   rate: {y_rate:.4f}  (should be ~0.13 from Exp1)")
            print(f"  out rate: {out_rate:.4f}  tgt: {tgt_rate:.4f}  non-tgt: {nontgt_rate:.4f}")
            print(f"  W_out before training: mean={model.l2mu_cell.W_out.weight.mean():.4f}  "
                  f"std={model.l2mu_cell.W_out.weight.std():.4f}")
            break
    print()

    # ── Phase 1: S2-STDP training ─────────────────────────────────────────
    print(f"{'='*60}")
    print("PHASE 1 — Supervised S2-STDP training of W_out")
    print(f"{'='*60}\n")

    best_train_acc = 0.0

    for epoch in range(num_epochs):

        model.train()

        sum_out_rate   = 0.0
        sum_tgt_rate   = 0.0
        sum_nontgt_rate = 0.0
        sum_error_tgt  = 0.0
        sum_error_nontgt = 0.0
        correct = 0
        total   = 0
        n_batches = 0

        for data, labels in train_loader:
            data   = data.to(device).swapaxes(0, 1)   # [T, B, input]
            labels = labels.to(device)

            with torch.no_grad():
                spk_out, spk_y = model(data)           # [T, B, 54]

            # Rates for this batch
            T = data.shape[0]
            B = labels.shape[0]
            spike_counts = spk_out.sum(dim=0)          # [B, 54]
            rates = spike_counts / T                    # [B, 54]

            tgt_idx    = torch.arange(0, L2MUCell.NUM_OUT, 2, device=device)
            nontgt_idx = torch.arange(1, L2MUCell.NUM_OUT, 2, device=device)
            sum_tgt_rate    += rates[:, tgt_idx].mean().item()
            sum_nontgt_rate += rates[:, nontgt_idx].mean().item()
            sum_out_rate    += rates.mean().item()

            # Error magnitude diagnostics (before update)
            R_mean = rates.mean(dim=1, keepdim=True)
            g = L2MUCell.S2STDP_G
            tgt_neuron_idx = labels * L2MUCell.NEURONS_PER_CLS   # [B]
            batch_idx = torch.arange(B, device=device)
            # target error = rate of target neuron − desired
            tgt_rates_per_sample = rates[batch_idx, tgt_neuron_idx]
            tgt_desired = (R_mean[:, 0] + (NUM_CLASSES - 1) / NUM_CLASSES * g).clamp(0, 1)
            sum_error_tgt += (tgt_rates_per_sample - tgt_desired).abs().mean().item()

            # non-target: mean over all non-target neurons
            nontgt_desired = (R_mean[:, 0] - g / NUM_CLASSES).clamp(0, 1)
            sum_error_nontgt += (rates[:, nontgt_idx].mean(dim=1) - nontgt_desired).abs().mean().item()

            # S2-STDP weight update
            model.l2mu_cell.s2stdp_update(labels)

            # Predictions (target neurons only)
            target_counts = spike_counts[:, tgt_idx]  # [B, 27]
            preds = target_counts.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += B
            n_batches += 1

        # ── Epoch summary ─────────────────────────────────────────────
        train_acc = correct / total * 100
        best_train_acc = max(best_train_acc, train_acc)
        W_mean = model.l2mu_cell.W_out.weight.mean().item()
        W_std  = model.l2mu_cell.W_out.weight.std().item()
        W_pos  = (model.l2mu_cell.W_out.weight > 0).float().mean().item()

        print(
            f"Ep {epoch+1:3d}/{num_epochs} | "
            f"acc: {train_acc:5.1f}% | "
            f"rate tgt/nontgt: {sum_tgt_rate/n_batches:.3f}/{sum_nontgt_rate/n_batches:.3f} | "
            f"|err| tgt/nontgt: {sum_error_tgt/n_batches:.4f}/{sum_error_nontgt/n_batches:.4f} | "
            f"W mean: {W_mean:+.4f}  std: {W_std:.4f}  pos%: {W_pos*100:.1f}%"
        )

    # ── Phase 2: Evaluation ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 2 — Evaluation on test set")
    print(f"{'='*60}\n")

    model.eval()

    correct = 0
    total   = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for data, labels in test_loader:
            data   = data.to(device).swapaxes(0, 1)
            labels = labels.to(device)

            spk_out, _ = model(data)
            spike_counts = spk_out.sum(dim=0)
            tgt_idx = torch.arange(0, L2MUCell.NUM_OUT, 2, device=device)
            preds = spike_counts[:, tgt_idx].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
            for lbl, pred in zip(labels.cpu(), preds.cpu()):
                class_total[lbl]   += 1
                class_correct[lbl] += int(pred == lbl)

    accuracy = correct / total * 100
    print(f"Test accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Best train accuracy: {best_train_acc:.2f}%\n")
    print("Per-class accuracy:")
    for c in range(NUM_CLASSES):
        n = class_total[c].item()
        acc_c = (class_correct[c] / n * 100) if n > 0 else float('nan')
        print(f"  Class {c:2d}: {int(class_correct[c].item()):3d}/{int(n):3d} = {acc_c:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / 'model_exp2b.pt')
    print(f"\nModel saved to: {save_path / 'model_exp2b.pt'}")

    return model, accuracy


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'order': 4.0,
        'theta': 7.0,
        'memory_size': 250.0,
        # m population (unchanged from Exp1)
        'beta_spk_m': 0.35,
        'threshold_spk_m': 0.4,
        # y population (unchanged from Exp1)
        'beta_spk_y': 0.3,
        'threshold_spk_y': 2.0,
        # out population
        # Threshold calibration:
        #   y rate ≈ 0.17, memory_size=250, W_out init ∈ [-0.1, 0.1]
        #   Expected input per step ≈ 0.17 * 250 * E[|w|] ≈ 0.17*250*0.05 ≈ 2.1
        #   LIF with beta=0.3 decays membrane fast; effective integrated
        #   current ≈ input / (1 - beta) ≈ 2.1 / 0.7 ≈ 3.0 at steady state.
        #   But many weights are negative, so net input is much less.
        #   Observed: out_rate collapses to ~0.003 with threshold=1.0.
        #   Lower to 0.5 to ensure sustained firing from epoch 1.
        'beta_spk_out': 0.3,
        'threshold_spk_out': 0.5,
    }

    train_exp2b(
        params=params,
        num_epochs=150,
        data_dir="../data/braille_full_splitted",
        split=2,
        D_weights_path="../model_insights/results/exp1/D_weights.pt",
        save_dir="../model_insights/results/exp2b",
    )