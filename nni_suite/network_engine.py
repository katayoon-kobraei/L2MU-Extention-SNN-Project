import lightning as pl
import torch
import torch.nn as nn
from architecture.full_precision.models.snn.l2mu import L2MU


class NetworkEngine(pl.LightningModule):
    def __init__(self, num_inputs, num_outputs, architecture, params):
        super().__init__()
        self.architecture = architecture
        self.model = L2MU(input_size=num_inputs, output_size=num_outputs,
                          params=params, neuron_type='Leaky')
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = params["lr"]
        self.save_hyperparameters()

        # Manual optimization so we control exactly when backprop and STDP run
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Only parameters with requires_grad=True (D and W_out excluded — they use STDP)
        backprop_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(backprop_params, lr=self.lr, betas=(0.9, 0.999))

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        train_data, train_labels = batch
        train_data = train_data.swapaxes(1, 0)

        # Step 1: forward pass (saves spike history for STDP, no weight changes)
        pred_output = self.model(train_data)

        # Step 2: backprop for Adam parameters (e_x, betas, thresholds)
        loss = self.loss_fn(pred_output.sum(0), train_labels)
        self.manual_backward(loss)
        optimizer.step()

        # Step 3: STDP update for D and W_out — runs AFTER backward, no graph conflict
        self.model.l2mu_cell.stdp_update()

        self.log("train_accuracy", self.calc_accuracy(pred_output, train_labels), prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        validation_data, validation_labels = batch
        validation_data = validation_data.swapaxes(1, 0)
        pred_output = self.model(validation_data)
        self.log("val_accuracy", self.calc_accuracy(pred_output, validation_labels), prog_bar=True)
        val_loss = self.loss_fn(pred_output.sum(0), validation_labels)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)
        pred_output = self.model(test_data)
        self.log("test_accuracy", self.calc_accuracy(pred_output, test_labels), prog_bar=True)
        test_loss = self.loss_fn(pred_output.sum(0), test_labels)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)
        pred_output = self.model(test_data)
        _, pred = pred_output.sum(dim=0).max(1)
        return pred.detach().cpu().numpy()

    @staticmethod
    def calc_accuracy(output, labels):
        _, idx = output.sum(dim=0).max(1)
        label_count = (labels == idx).sum()
        return label_count / len(labels)