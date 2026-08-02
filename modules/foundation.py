import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


_MIN_COVERAGE_RATIO = 0.80


class RETFoundDINO(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2",
        num_classes: int = 39,
        projection_dim: int = 128,
        checkpoint_path: str = "weights/RETFound_dinov2_shanghai.pth",
        image_size: int = 518,
    ) -> None:
        """
        RETFound DINOv2 wrapper.

        Args:
            model_name:      timm architecture string.  Must be ViT-L/14 for the
                             RETFound_DINOv2 checkpoint (patch size 14, not 16).
            num_classes:     Number of output classes for the classifier head.
            projection_dim:  Output dimension of the projection head (128 per spec).
            checkpoint_path: Path to the pretrained .pth file.
            image_size:      Input image size (square).
        """
        super().__init__()

        print("Initialising backbone: %s" % (model_name,))
        self.image_size = image_size
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
        )
        self.embed_dim: int = self.backbone.num_features
        self.num_classes = num_classes


        self.classification_head = nn.Linear(self.embed_dim, num_classes)

        self.projection_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

        if checkpoint_path:
            self._load_backbone_checkpoint(checkpoint_path)

        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)


    def _load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load backbone weights from a DINOv2 teacher-student checkpoint.

        Key lookup order:
          1. "teacher"    — standard DINOv2 teacher-student format
          2. "model"      — generic timm/RETFound wrapper
          3. "state_dict" — generic PyTorch Lightning / HF wrapper
          4. bare dict    — raw state dict (fails loudly if coverage is low)

        Raises
        ------
        FileNotFoundError
            If checkpoint_path does not exist on disk.
        RuntimeError
            If loaded backbone weight coverage is below _MIN_COVERAGE_RATIO,
            which almost always means the wrong file was supplied.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print("Loading checkpoint: %s" % (checkpoint_path,))
        raw = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(raw, dict):
            top_keys = list(raw.keys())
            print("Checkpoint top-level keys: %s" % (top_keys,))

            if "teacher" in raw:
                state_dict = raw["teacher"]
                print("Using 'teacher' sub-dict from checkpoint.")
            elif "model" in raw:
                state_dict = raw["model"]
                print("Using 'model' sub-dict from checkpoint.")
            elif "state_dict" in raw:
                state_dict = raw["state_dict"]
                print("Using 'state_dict' sub-dict from checkpoint.")
            else:
                print("WARNING: No known wrapper key found in checkpoint. Treating entire dict as state dict. Top-level keys: %s" % (top_keys,))
                state_dict = raw
        else:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} is not a dict "
                f"(got {type(raw).__name__}). Cannot load weights."
            )


        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("backbone.", "")
            cleaned[k] = v

        backbone_dict = self.backbone.state_dict()
        matched = {
            k: v
            for k, v in cleaned.items()
            if k in backbone_dict and v.shape == backbone_dict[k].shape
        }
        coverage = len(matched) / max(len(backbone_dict), 1)

        print("Backbone weight coverage: %d / %d keys (%.1f%%)" % (len(matched), len(backbone_dict), coverage * 100))

        if coverage < _MIN_COVERAGE_RATIO:
            raise RuntimeError(
                f"Backbone weight coverage is only {coverage:.1%} "
                f"({len(matched)}/{len(backbone_dict)} keys matched). "
                "This almost certainly means the wrong checkpoint file was "
                "supplied. Check that the path in your config points to the "
                "RETFound_DINOv2 checkpoint, not a different model. "
                f"Checkpoint top-level keys were: {list(raw.keys()) if isinstance(raw, dict) else 'N/A'}"
            )

        missing, unexpected = self.backbone.load_state_dict(matched, strict=False)
        print("Backbone weights loaded successfully.")
        if missing:
            print("Missing keys (not in checkpoint): %d" % (len(missing),))
        if unexpected:
            print("Unexpected keys (not in backbone): %d" % (len(unexpected),))


    def unfreeze_last_n_blocks(self, num_layers: int) -> None:
        """
        Freeze the entire network, then selectively unfreeze:
          - The classifier head
          - The projection head
          - The last `num_layers` transformer blocks of the backbone
          - The backbone's final norm layer   ← required by spec §4

        Args:
            num_layers: Number of trailing transformer blocks to unfreeze.
        """
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classification_head.parameters():
            param.requires_grad = True
        for param in self.projection_head.parameters():
            param.requires_grad = True

        if not hasattr(self.backbone, "blocks"):
            print("WARNING: Backbone has no 'blocks' attribute — cannot selectively unfreeze transformer blocks.")
            return

        total_blocks = len(self.backbone.blocks)
        n = min(max(0, num_layers), total_blocks)
        start_idx = total_blocks - n

        print("Unfreezing last %d / %d backbone blocks (indices %d–%d) + classification head + projection head + final norm." % (n, total_blocks, start_idx, total_blocks - 1))

        for i in range(start_idx, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print("Final backbone norm layer unfrozen.")
        else:
            print("WARNING: backbone.norm not found — final norm layer not unfrozen.")

    def count_parameters(self) -> dict[str, float]:
        """Return trainable / total parameter counts (in millions) per component."""
        def _count(module: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        bb_total, bb_train = _count(self.backbone)
        cls_total, _ = _count(self.classification_head)
        proj_total, _ = _count(self.projection_head)
        all_total, all_train = _count(self)

        stats = {
            "backbone_total_M": bb_total / 1e6,
            "backbone_trainable_M": bb_train / 1e6,
            "classification_head_total_M": cls_total / 1e6,
            "projection_head_total_M": proj_total / 1e6,
            "model_total_M": all_total / 1e6,
            "model_trainable_M": all_train / 1e6,
        }

        print("Parameters — backbone: %.2fM total / %.2fM trainable | cls head: %.2fM | proj head: %.2fM | model total: %.2fM / trainable: %.2fM" % (
            stats["backbone_total_M"], stats["backbone_trainable_M"], stats["classification_head_total_M"], stats["projection_head_total_M"], stats["model_total_M"], stats["model_trainable_M"],
        ))
        return stats


    def forward(
        self,
        x: torch.Tensor,
        mode: str = "classification",
        use_temperature: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               Image tensor [B, C, H, W].
            mode:             One of "classification", "projection", "features", "both".
            use_temperature:  If True (default), classification logits are
                              divided by self.temperature before being
                              returned. No-op while temperature == 1.0 (i.e.
                              before tune_temperature() has been run), so this
                              does not change training dynamics.

        Returns:
            "classification" → logits [B, num_classes]
            "projection"     → L2-normalised projection [B, projection_dim]
            "features"       → raw backbone features [B, embed_dim]
            "both"           → (logits, L2-normalised projection)
        """
        features = self.backbone(x)

        if mode == "classification":
            logits = self.classification_head(features)
            if use_temperature:
                logits = logits / self.temperature
            return logits

        if mode == "projection":
            proj = self.projection_head(features)
            return F.normalize(proj, dim=1)

        if mode == "features":
            return features

        if mode == "both":
            logits = self.classification_head(features)
            if use_temperature:
                logits = logits / self.temperature
            proj = F.normalize(self.projection_head(features), dim=1)
            return logits, proj

        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Choose 'classification', 'projection', 'features', or 'both'."
        )