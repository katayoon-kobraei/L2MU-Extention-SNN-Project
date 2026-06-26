"""
Experiment 2D: Backpropagation baseline for the frozen SSM backbone.

Architecture:
    - m + y populations: frozen (D and W_in loaded from Exp1)
    - W_out [250 → 27]: trained with cross-entropy loss + Adam
    - No WTA, no STDP

Purpose:
    Establish the ceiling accuracy achievable with these specific D and W_in
    weights when the readout is given gradient-based learning.
    This result is the fair upper bound for all STDP variants (Exp2A/B/C)
    that use the same frozen backbone.

Training design:
    The backbone runs with torch.no_grad() — no BPTT through 256 timesteps.
    W_out sees spike rates (mean over T) as input, outputs raw logits.
    Gradient flows only through the single linear layer W_out.
    This is equivalent to training a logistic regression on frozen spike features.
"""

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn.functional as F
from pathlib import Path
from lightning.pytorch import seed_everything
from data.BRAILLE import BRAILLE
from architecture.full_precision.models.snn.l2mu import L2MU
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell

torch.set_float32_matmul_precision('high')

NUM_CLASSES = 27


def train_exp2d(
    params,
    num_epochs=150,
    lr=1e-3,
    data_dir="data/braille_full_splitted",
    split=2,
    D_weights_path="model_insights/results/exp1/D_weights.pt",
    W_in_weights_path="model_insights/results/exp1/W_in_weights.pt",
    save_dir="model_insights/results/exp2d",
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
    model.l2mu_cell.load_W_in_from_exp1(W_in_weights_path)

    # ── Optimizer: only W_out.weight ──────────────────────────────────────
    # Everything else is frozen — D, W_in, A, B, C, LIF params.
    optimizer = torch.optim.Adam([model.l2mu_cell.W_out.weight], lr=lr)

    # Verify only W_out is trainable
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    print(f"\nTrainable parameters:")
    for name, shape in trainable:
        print(f"  {name}: {list(shape)}")
    print(f"  Total: {sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)} params")
    print(f"\nLearning rate    : {lr}")
    print(f"threshold_spk_y  : {params['threshold_spk_y']}")

    # ── Sanity check ──────────────────────────────────────────────────────
    print("\n── Sanity check: y-population spike rate (first batch) ──")
    model.eval()
    with torch.no_grad():
        for data, _ in train_loader:
            data   = data.to(device).swapaxes(0, 1)
            logits, spk_y_stack = model(data)
            y_rate = spk_y_stack.mean().item()
            print(f"  y rate : {y_rate:.4f}   (should be ~0.098)")
            print(f"  logits : mean={logits.mean():.4f}  std={logits.std():.4f}")
            break
    print()

    # ── Training ──────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("PHASE 1 — Cross-entropy + Adam training of W_out")
    print(f"{'='*60}\n")

    best_train_acc = 0.0

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0
        n_batches  = 0

        for data, labels in train_loader:
            data   = data.to(device).swapaxes(0, 1)   # [T, B, 24]
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward: backbone runs with no_grad inside model
            logits, _ = model(data)                    # [B, 27]

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.shape[0]
            n_batches  += 1

        train_acc      = correct / total * 100
        best_train_acc = max(best_train_acc, train_acc)
        avg_loss       = total_loss / n_batches
        W_mean         = model.l2mu_cell.W_out.weight.mean().item()
        W_std          = model.l2mu_cell.W_out.weight.std().item()

        print(
            f"Ep {epoch+1:3d}/{num_epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"train acc: {train_acc:5.1f}% | "
            f"W mean: {W_mean:+.4f}  std: {W_std:.4f}"
        )

    # ── Evaluation ────────────────────────────────────────────────────────
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

            logits, _ = model(data)
            preds     = logits.argmax(dim=1)

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
        print(f"  Class {c:2d}: {int(class_correct[c].item()):2d}/{int(n):2d} = {acc_c:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / 'model_exp2d.pt')
    print(f"\nModel saved to: {save_path / 'model_exp2d.pt'}")

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
        # y population — must match Exp1 (threshold=7.0 → rate~0.098)
        'beta_spk_y': 0.3,
        'threshold_spk_y': 7.0,
        # No spk_out population — W_out outputs logits directly
    }

    train_exp2d(
        params=params,
        num_epochs=150,
        lr=1e-3,
        data_dir="data/braille_full_splitted",
        split=2,
        D_weights_path="model_insights/results/exp1/D_weights.pt",
        W_in_weights_path="model_insights/results/exp1/W_in_weights.pt",
        save_dir="model_insights/results/exp2d",
    )
