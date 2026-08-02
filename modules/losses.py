
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-balanced focal loss using the effective-number-of-samples formulation.

    Per-class weight:
        w_c = (1 - beta) / (1 - beta^n_c)
    normalised so that sum(w) = num_classes.

    Focal modulation:  loss = w_c * (1 - p_t)^gamma * CE(logits, target)

    Args:
        samples_per_class: List/tensor of sample counts per class.
        beta:              Smoothing factor (0 = no re-weighting, → 1 = inverse
                           frequency).  0.9999 is standard.
        gamma:             Focal exponent.  2.0 is standard.
        reduction:         "mean" | "sum" | "none".
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        spc = torch.tensor(samples_per_class, dtype=torch.float32)
        present = spc > 0
        if not torch.any(present):
            raise ValueError("ClassBalancedFocalLoss requires at least one sample.")

        effective_num = 1.0 - torch.pow(beta, spc[present])
        present_weights = (1.0 - beta) / effective_num
        weights = torch.zeros_like(spc)
        weights[present] = present_weights
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.reduction = reduction
        print("ClassBalancedFocalLoss — beta=%.4f, gamma=%.1f, weight range [%.3f, %.3f]" % (beta, gamma, weights.min().item(), weights.max().item()))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.class_weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al., 2020).

    Expects L2-normalised feature vectors (the projection head in forward.py
    applies F.normalize before returning, so this is already satisfied).

    Args:
        temperature: Logit scale.  0.07 matches the SupCon paper default for
                     supervised settings with many classes.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalised projections [B, D].
            labels:   Integer class labels [B].
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature

        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        self_mask = torch.eye(batch_size, device=device)
        logits_mask = 1.0 - self_mask
        mask = mask * logits_mask   # positives exclude self

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(dim=1)
        pos_per_anchor = torch.where(
            pos_per_anchor < 1e-6,
            torch.ones_like(pos_per_anchor),
            pos_per_anchor,
        )

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_anchor
        return -mean_log_prob_pos.mean()


class CombinedLoss(nn.Module):
    """
    L = L_focal(logits, labels) + lam * L_supcon(proj, labels)

    Returns a tuple (total, focal_detached, supcon_detached) so the trainer
    can log each component separately without retaining extra graphs.

    Args:
        focal_loss:  Instantiated ClassBalancedFocalLoss.
        supcon_loss: Instantiated SupConLoss.
        lam:         Contrastive loss weight.  0.0 → focal-only ablation.
                     0.2 → combined run (spec §5).
    """

    def __init__(
        self,
        focal_loss: ClassBalancedFocalLoss,
        supcon_loss: SupConLoss,
        lam: float = 0.2,
    ) -> None:
        super().__init__()
        self.focal_loss = focal_loss
        self.supcon_loss = supcon_loss
        self.lam = lam

    def forward(
        self,
        logits: torch.Tensor,
        proj: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_focal = self.focal_loss(logits, labels)
        l_supcon = self.supcon_loss(proj, labels)
        total = l_focal + self.lam * l_supcon
        return total, l_focal.detach(), l_supcon.detach()