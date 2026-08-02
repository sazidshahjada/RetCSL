import os
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm


class FoundationModelTrainer:
    """
    Trainer for the combined-loss fine-tuning pilot.

    Args:
        model:                    RETFoundDINO instance (already has
                                  unfreeze_last_n_blocks() called before being
                                  passed here).
        train_loader_contrastive: P-K sampled DataLoader.  This is the primary
                                  training loader — it guarantees valid positive
                                  pairs for SupConLoss every batch.
        val_loader:               Standard (non-P-K) validation DataLoader.
        criterion:                CombinedLoss instance.
        optimizer:                AdamW initialised only on trainable params.
        device:                   torch.device.
        checkpoint_path:          Path to write the best-F1 checkpoint.
        patience:                 Early-stopping patience (epochs).
        max_grad_norm:            Gradient clipping max norm.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader_contrastive,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_path: str = "results/checkpoints/best_model.pth",
        patience: int = 7,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader_contrastive
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.max_grad_norm = max_grad_norm

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        self.scaler = torch.cuda.amp.GradScaler()

        self.history: dict[str, list[float]] = {
            "train_loss_total": [],
            "train_loss_focal": [],
            "train_loss_supcon": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)


    def train_one_epoch(self, epoch: int, num_epochs: int) -> tuple[float, float, float]:
        """
        One full pass over train_loader (P-K sampled).

        Returns:
            (mean_total_loss, mean_focal_loss, mean_supcon_loss)
        """
        self.model.train()

        total_loss_sum = 0.0
        focal_loss_sum = 0.0
        supcon_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train - Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, proj = self.model(images, mode="both")
                total, l_focal, l_supcon = self.criterion(logits, proj, labels)

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += total.item()
            focal_loss_sum += l_focal.item()
            supcon_loss_sum += l_supcon.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{total.item():.4f}",
                focal=f"{l_focal.item():.4f}",
                supcon=f"{l_supcon.item():.4f}",
            )

        return (
            total_loss_sum / n_batches,
            focal_loss_sum / n_batches,
            supcon_loss_sum / n_batches,
        )


    def validate(self, epoch: int, num_epochs: int) -> tuple[float, float, float]:
        """
        Validation pass.

        Returns:
            (mean_val_loss, accuracy, macro_f1)
        """
        self.model.eval()

        total_loss_sum = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        n_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Val - Epoch {epoch + 1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    logits, proj = self.model(images, mode="both")
                    total, _, _ = self.criterion(logits, proj, labels)

                total_loss_sum += total.item()
                n_batches += 1

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = total_loss_sum / n_batches
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return val_loss, accuracy, macro_f1


    def fit(self, num_epochs: int = 100) -> tuple[nn.Module, dict]:
        """
        Train for up to num_epochs with early stopping on validation macro-F1.

        Checkpoint criterion: macro-F1 (spec §7.6).  Using F1 rather than loss
        is more meaningful for a 39-class imbalanced dataset, and ensures the
        lambda=0.2 and lambda=0.0 runs are compared on the same criterion.

        Returns:
            (best_model, history_dict)
        """
        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(num_epochs):
            tr_total, tr_focal, tr_supcon = self.train_one_epoch(epoch, num_epochs)
            val_loss, val_acc, val_f1 = self.validate(epoch, num_epochs)
            self.scheduler.step()

            self.history["train_loss_total"].append(tr_total)
            self.history["train_loss_focal"].append(tr_focal)
            self.history["train_loss_supcon"].append(tr_supcon)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            print("Epoch [%d/%d] | train_loss=%.4f (focal=%.4f supcon=%.4f) | val_loss=%.4f val_acc=%.4f val_f1=%.4f" % (
                epoch + 1, num_epochs, tr_total, tr_focal, tr_supcon, val_loss, val_acc, val_f1,
            ))
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"train={tr_total:.4f} (f={tr_focal:.4f} sc={tr_supcon:.4f}) | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved at epoch {epoch + 1} with macro-F1={best_val_f1:.4f}.\n")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping at epoch %d (patience=%d)." % (epoch + 1, self.patience))
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print("Best weights reloaded from %s." % (self.checkpoint_path,))

        return self.model, self.history


    def evaluate(
        self,
        test_loader,
        class_names: Optional[list[str]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Evaluate on test_loader.  Loads from checkpoint_path (or the trainer's
        default checkpoint) before running.

        Args:
            test_loader:     DataLoader for the test split.
            class_names:     Optional list of class name strings for the report.
            checkpoint_path: Override checkpoint path.

        Returns:
            Dict with accuracy, macro_f1, and avg_confidence keys.
        """
        path = checkpoint_path or self.checkpoint_path
        if path and os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            print("Loaded weights for evaluation from %s" % (path,))

        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_confidences: list[float] = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    logits = self.model(images, mode="classification")
                preds = logits.argmax(dim=1)
                confidences = logits.softmax(dim=1).max(dim=1).values
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_confidences.extend(confidences.cpu().numpy().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        avg_confidence = sum(all_confidences) / len(all_confidences)

        print("Test accuracy=%.4f  macro_F1=%.4f  avg_confidence=%.4f" % (accuracy, macro_f1, avg_confidence))
        print(f"\nTest accuracy: {accuracy:.4f}  macro-F1: {macro_f1:.4f}  avg_confidence: {avg_confidence:.4f}")
        print(
            classification_report(
                all_labels, all_preds,
                target_names=class_names,
                zero_division=0,
            )
        )

        return {"accuracy": accuracy, "macro_f1": macro_f1, "avg_confidence": avg_confidence}


    def tune_temperature(
        self,
        val_loader,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Post-hoc temperature scaling (Guo et al., 2017), run once after fit()
        has reloaded the best checkpoint.

        Collects raw (unscaled) logits and labels over val_loader, then fits
        a single scalar temperature T by minimising NLL with LBFGS. Only
        self.model.temperature is optimised — every other parameter stays
        frozen, and no gradients flow into the backbone or heads.

        Must be called after fit() and before evaluate(), and the caller is
        responsible for re-persisting the checkpoint afterward (see the
        training script) so that a later evaluate() reload doesn't discard
        the fitted value.

        Returns:
            Fitted temperature value (float).
        """
        self.model.eval()

        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Collecting logits for temperature scaling"):
                images = images.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    logits = self.model(images, mode="classification", use_temperature=False)
                logits_list.append(logits.float().cpu())
                labels_list.append(labels)

        all_logits = torch.cat(logits_list).to(self.device)
        all_labels = torch.cat(labels_list).to(self.device)

        nll_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            pre_nll = nll_criterion(all_logits, all_labels).item()

        self.model.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.model.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(all_logits / self.model.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.temperature.requires_grad = False

        with torch.no_grad():
            post_nll = nll_criterion(all_logits / self.model.temperature, all_labels).item()

        t_value = self.model.temperature.item()
        print(
            "Temperature scaling: T=%.4f | val NLL before=%.4f after=%.4f"
            % (t_value, pre_nll, post_nll)
        )
        return t_value