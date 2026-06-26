"""
Experiment 2C: S2-STDP + WTA on the y population.

Identical to Exp2B except:
    1. W_in loaded from Exp1 (so m population sees the same u_t as during STDP training)
    2. WTA applied in the cell after spk_y (handled in l2mu_cell_exp2c.py)
    3. Results saved to model_insights/results/exp2c/

Architecture:
    - m + y populations: frozen (D and W_in loaded from Exp1)
    - y population fires → WTA mask → only winner passes to W_out
    - out population: 54 LIF neurons (2 per Braille class)
    - W_out trained with rate-code S2-STDP (supervised, labels used per batch)
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


def train_exp2c(
    params,
    num_epochs=150,
    data_dir="data/braille_full_splitted",
    split=2,
    D_weights_path="model_insights/results/exp1/D_weights.pt",
    W_in_weights_path="model_insights/results/exp1/W_in_weights.pt",   # ← NEW
    save_dir="model_insights/results/exp2c",                            # ← changed
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

    # Load frozen weights from Exp1
    model.l2mu_cell.load_D_from_exp1(D_weights_path)
    model.l2mu_cell.load_W_in_from_exp1(W_in_weights_path)   # ← NEW

    print(f"\nW_out shape     : {model.l2mu_cell.W_out.weight.shape}")
    print(f"Output neurons  : {L2MUCell.NUM_OUT} "
          f"({L2MUCell.NEURONS_PER_CLS} per class × {NUM_CLASSES} classes)")
    print(f"threshold_spk_y  : {params['threshold_spk_y']}")
    print(f"threshold_spk_out: {params['threshold_spk_out']}")
    print(f"R_TARGET         : {L2MUCell.R_TARGET}")
    print(f"R_NON_TARGET     : {L2MUCell.R_NON_TARGET}")
    print(f"S2STDP_A_LR      : {L2MUCell.S2STDP_A_LR}")
    print(f"S2STDP_A_LR_ERR  : {L2MUCell.S2STDP_A_LR_ERR}")
    print(f"WTA              : enabled")

    # ── Sanity check: spike rates before training ─────────────────────────
    print("\n── Sanity check: spike rates (first batch, before training) ──")
    model.eval()
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device).swapaxes(0, 1)
            spk_out, spk_y = model(data)
            T = data.shape[0]
            y_rate      = spk_y.sum(dim=0).mean().item() / T
            out_rate    = spk_out.sum(dim=0).mean().item() / T
            tgt_rate    = spk_out.sum(dim=0)[:, ::2].mean().item() / T
            nontgt_rate = spk_out.sum(dim=0)[:, 1::2].mean().item() / T
            print(f"  y   rate (raw, pre-WTA)  : {y_rate:.4f}  (should be ~0.098 from Exp1)")
            print(f"  out rate: {out_rate:.4f}  tgt: {tgt_rate:.4f}  non-tgt: {nontgt_rate:.4f}")
            print(f"  W_out before training: mean={model.l2mu_cell.W_out.weight.mean():.4f}  "
                  f"std={model.l2mu_cell.W_out.weight.std():.4f}")
            break
    print()

    # ── Phase 1: S2-STDP training ─────────────────────────────────────────
    print(f"{'='*60}")
    print("PHASE 1 — Supervised S2-STDP + WTA training of W_out")
    print(f"{'='*60}\n")

    best_train_acc = 0.0

    for epoch in range(num_epochs):

        model.train()

        sum_out_rate     = 0.0
        sum_tgt_rate     = 0.0
        sum_nontgt_rate  = 0.0
        sum_error_tgt    = 0.0
        sum_error_nontgt = 0.0
        correct   = 0
        total     = 0
        n_batches = 0

        for data, labels in train_loader:
            data   = data.to(device).swapaxes(0, 1)   # [T, B, input]
            labels = labels.to(device)

            with torch.no_grad():
                spk_out, spk_y = model(data)           # [T, B, 54], [T, B, 250]

            # Rates for this batch
            T = data.shape[0]
            B = labels.shape[0]
            spike_counts = spk_out.sum(dim=0)          # [B, 54]
            rates        = spike_counts / T             # [B, 54]

            tgt_idx    = torch.arange(0, L2MUCell.NUM_OUT, 2, device=device)
            nontgt_idx = torch.arange(1, L2MUCell.NUM_OUT, 2, device=device)
            sum_tgt_rate    += rates[:, tgt_idx].mean().item()
            sum_nontgt_rate += rates[:, nontgt_idx].mean().item()
            sum_out_rate    += rates.mean().item()

            # Error magnitude diagnostics (before update)
            tgt_neuron_idx       = labels * L2MUCell.NEURONS_PER_CLS   # [B]
            batch_idx            = torch.arange(B, device=device)
            tgt_rates_per_sample = rates[batch_idx, tgt_neuron_idx]

            sum_error_tgt    += (tgt_rates_per_sample - L2MUCell.R_TARGET).abs().mean().item()
            sum_error_nontgt += (rates[:, nontgt_idx].mean(dim=1) - L2MUCell.R_NON_TARGET).abs().mean().item()

            # S2-STDP weight update (uses spk_y_wta accumulated inside cell)
            model.l2mu_cell.s2stdp_update(labels)

            # Predictions (target neurons only)
            target_counts = spike_counts[:, tgt_idx]   # [B, 27]
            preds         = target_counts.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += B
            n_batches += 1

        # ── Epoch summary ─────────────────────────────────────────────────
        train_acc      = correct / total * 100
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

    correct       = 0
    total         = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for data, labels in test_loader:
            data   = data.to(device).swapaxes(0, 1)
            labels = labels.to(device)

            spk_out, _ = model(data)
            spike_counts = spk_out.sum(dim=0)
            tgt_idx      = torch.arange(0, L2MUCell.NUM_OUT, 2, device=device)
            preds        = spike_counts[:, tgt_idx].argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
            for lbl, pred in zip(labels.cpu(), preds.cpu()):
                class_total[lbl]   += 1
                class_correct[lbl] += int(pred == lbl)

    accuracy = correct / total * 100
    print(f"Test accuracy       : {correct}/{total} = {accuracy:.2f}%")
    print(f"Best train accuracy : {best_train_acc:.2f}%\n")
    print("Per-class accuracy:")
    for c in range(NUM_CLASSES):
        n     = class_total[c].item()
        acc_c = (class_correct[c] / n * 100) if n > 0 else float('nan')
        print(f"  Class {c:2d}: {int(class_correct[c].item()):3d}/{int(n):3d} = {acc_c:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / 'model_exp2c.pt')   # ← changed
    print(f"\nModel saved to: {save_path / 'model_exp2c.pt'}")

    return model, accuracy


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'order': 4.0,
        'theta': 7.0,
        'memory_size': 250.0,
        # m population — unchanged from Exp1
        'beta_spk_m': 0.35,
        'threshold_spk_m': 0.4,
        # y population — best Exp2B config
        'beta_spk_y': 0.3,
        'threshold_spk_y': 7.0,    # best Exp2B value
        # out population — best Exp2B config
        'beta_spk_out': 0.3,
        'threshold_spk_out': 0.05,
    }

    train_exp2c(
        params=params,
        num_epochs=150,
        data_dir="data/braille_full_splitted",
        split=2,
        D_weights_path="model_insights/results/exp1/D_weights.pt",
        W_in_weights_path="model_insights/results/exp1/W_in_weights.pt",
        save_dir="model_insights/results/exp2c",
    )
