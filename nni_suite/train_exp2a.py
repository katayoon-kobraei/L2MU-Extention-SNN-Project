"""
Experiment 2A: Label assignment output layer (Diehl & Cook style).

Architecture:
    - m + y populations: frozen (D loaded from Exp1)
    - out population: 27 LIF neurons (one per Braille class)
    - W_out trained with unsupervised STDP (no labels during training)

After training:
    Phase 1 — STDP training of W_out (no labels)
    Phase 2 — Label assignment (one pass over training set WITH labels)
    Phase 3 — Evaluation on test set (accuracy reported)
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

NUM_CLASSES = 27


def train_exp2a(params, num_epochs=150,
                data_dir="data/braille_full_splitted",
                split=2,
                D_weights_path="model_insights/results/exp1/D_weights.pt",
                save_dir="model_insights/results/exp2a"):

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
    data_module.setup('test')
    train_loader = data_module.train_dataloader()
    test_loader  = data_module.test_dataloader()
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Test samples:     {len(data_module.test_dataset)}")

    # --- Model ---
    model = L2MU(
        input_size=data_module.num_inputs,
        params=params,
        neuron_type='Leaky',
    ).to(device)

    # Load D weights from Exp1 and freeze
    model.l2mu_cell.load_D_from_exp1(D_weights_path)

    print(f"\nW_out shape: {model.l2mu_cell.W_out.weight.shape}")
    print(f"\n{'='*60}")
    print("PHASE 1 — Unsupervised STDP training of W_out")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # PHASE 1: Train W_out with unsupervised STDP
    # ----------------------------------------------------------------
    for epoch in range(num_epochs):

        total_out_rate = 0.0
        num_batches    = 0

        for data, _ in train_loader:
            # Labels ignored — unsupervised!
            data = data.to(device).swapaxes(0, 1)   # [T, B, input_size]

            with torch.no_grad():
                spk_out, spk_y = model(data)         # [T, B, 27], [T, B, 250]

            total_out_rate += spk_out.mean().item()
            num_batches    += 1

            model.l2mu_cell.stdp_update()

        avg_out_rate = total_out_rate / num_batches
        W_mean = model.l2mu_cell.W_out.weight.mean().item()
        W_std  = model.l2mu_cell.W_out.weight.std().item()

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"out spike rate: {avg_out_rate:.4f} | "
            f"W_out mean: {W_mean:.4f} | "
            f"W_out std: {W_std:.4f}"
        )

    # ----------------------------------------------------------------
    # PHASE 2: Label assignment
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2 — Label assignment")
    print(f"{'='*60}\n")

    model.eval()

    # Count how many times each output neuron fired for each class
    # Shape: [27 output neurons, 27 classes]
    neuron_class_counts = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    with torch.no_grad():
        for data, labels in train_loader:
            data   = data.to(device).swapaxes(0, 1)   # [T, B, input_size]
            labels = labels.to(device)                  # [B]

            spk_out, _ = model(data)                   # [T, B, 27]

            # Sum spikes over time for each sample
            spike_counts = spk_out.sum(dim=0)          # [B, 27]

            for i, label in enumerate(labels):
                neuron_class_counts[:, label] += spike_counts[i].cpu()

    # Assign each neuron to the class it fired most for
    neuron_assignments = neuron_class_counts.argmax(dim=1)  # [27]

    # Check for dead neurons (never fired)
    neuron_total_spikes = neuron_class_counts.sum(dim=1)    # [27]
    dead_neurons = (neuron_total_spikes == 0).sum().item()

    print(f"Neuron assignments: {neuron_assignments.tolist()}")
    print(f"Dead neurons: {dead_neurons}/{NUM_CLASSES}")
    print(f"Unique classes covered: {len(neuron_assignments.unique())}/27")

    # ----------------------------------------------------------------
    # PHASE 3: Evaluation on test set
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3 — Evaluation on test set")
    print(f"{'='*60}\n")

    correct = 0
    total   = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data   = data.to(device).swapaxes(0, 1)
            labels = labels.to(device)

            spk_out, _ = model(data)                   # [T, B, 27]
            spike_counts = spk_out.sum(dim=0)          # [B, 27]

            # For each sample, compute average spike count per class
            # using neuron assignments
            for i, label in enumerate(labels):
                counts = spike_counts[i].cpu()          # [27]

                # Average firing per class
                class_votes = torch.zeros(NUM_CLASSES)
                class_neuron_counts = torch.zeros(NUM_CLASSES)

                for neuron_idx in range(NUM_CLASSES):
                    if neuron_total_spikes[neuron_idx] > 0:  # skip dead
                        assigned_class = neuron_assignments[neuron_idx].item()
                        class_votes[assigned_class]        += counts[neuron_idx]
                        class_neuron_counts[assigned_class] += 1

                # Normalize by number of neurons per class
                mask = class_neuron_counts > 0
                class_votes[mask] /= class_neuron_counts[mask]

                predicted = class_votes.argmax().item()
                if predicted == label.item():
                    correct += 1
                total += 1

    accuracy = correct / total * 100
    print(f"Test accuracy: {correct}/{total} = {accuracy:.2f}%")

    # --- Save ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / 'model_exp2a.pt')
    torch.save(neuron_assignments, save_path / 'neuron_assignments.pt')
    print(f"\nModel saved to: {save_path / 'model_exp2a.pt'}")

    return model, neuron_assignments, accuracy


if __name__ == '__main__':

    params = {
        'batch_size': 64,
        'order': 4.0,
        'theta': 7.0,
        'memory_size': 250.0,
        'beta_spk_m': 0.35,
        'threshold_spk_m': 0.4,
        'beta_spk_y': 0.3,
        'threshold_spk_y': 2.0,
        'beta_spk_out': 0.3,
        'threshold_spk_out': 75.0,
    }

    train_exp2a(
        params=params,
        num_epochs=150,
        data_dir="../data/braille_full_splitted",
        split=2,
        D_weights_path="../model_insights/results/exp1/D_weights.pt",
        save_dir="../model_insights/results/exp2a",
    )
