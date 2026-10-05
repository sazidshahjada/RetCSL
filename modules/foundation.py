import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


_MIN_COVERAGE_RATIO = 0.80


class _FeatureBackbone(nn.Module):
    """Normalize model-specific encoders to a [batch, feature_dim] tensor."""

    def __init__(self, encoder: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_features = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.encoder(images)
        if isinstance(output, dict):
            for key in ("x_norm_clstoken", "last_hidden_state", "pooler_output"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim == 3:
            output = output[:, 0]
        if output.ndim > 2:
            output = output.flatten(1)
        return output


class FoundationClassifier(nn.Module):
    """Common classifier/projection heads for the supported image encoders.

    Every model returns logits [B, num_classes], a normalized projection
    [B, projection_dim], or raw encoder features via the same ``mode`` API as
    :class:`RETFoundDINO`. ``backbone`` is the actual image encoder, so callers
    can use its transformer blocks for partial fine-tuning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 39,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classification_head = nn.Linear(embed_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor, mode: str = "classification", use_temperature: bool = True):
        features = self.backbone(x)
        if mode == "features":
            return features
        if mode in ("classification", "both"):
            logits = self.classification_head(features)
            if use_temperature:
                logits = logits / self.temperature
        if mode == "classification":
            return logits
        if mode == "projection":
            return F.normalize(self.projection_head(features), dim=1)
        if mode == "both":
            return logits, F.normalize(self.projection_head(features), dim=1)
        raise ValueError("mode must be 'classification', 'projection', 'features', or 'both'")

    def unfreeze_last_n_blocks(self, num_layers: int) -> None:
        """Freeze the encoder, then unfreeze its final transformer blocks and heads."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for head in (self.classification_head, self.projection_head):
            for parameter in head.parameters():
                parameter.requires_grad = True
        encoder = getattr(self.backbone, "encoder", self.backbone)
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            transformer = getattr(encoder, "transformer", None)
            blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            raise AttributeError("This encoder exposes no transformer blocks for partial unfreezing")
        count = min(max(int(num_layers), 0), len(blocks))
        for block in list(blocks)[-count:] if count else []:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("norm", "ln_post"):
            norm = getattr(encoder, name, None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad = True

    def count_parameters(self) -> dict[str, float]:
        """Return trainable and total parameter counts per model component."""
        def count(module: nn.Module) -> tuple[int, int]:
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            return total, trainable

        backbone_total, backbone_trainable = count(self.backbone)
        classifier_total, _ = count(self.classification_head)
        projection_total, _ = count(self.projection_head)
        model_total, model_trainable = count(self)
        return {
            "backbone_total_M": backbone_total / 1e6,
            "backbone_trainable_M": backbone_trainable / 1e6,
            "classification_head_total_M": classifier_total / 1e6,
            "projection_head_total_M": projection_total / 1e6,
            "model_total_M": model_total / 1e6,
            "model_trainable_M": model_trainable / 1e6,
        }


def _load_encoder_checkpoint(encoder: nn.Module, path: str, min_coverage: float = 0.8) -> None:
    """Load a local encoder checkpoint, accepting common wrapper prefixes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "teacher", "model_state_dict"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not isinstance(raw, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a state dict")
    target = encoder.state_dict()
    matched = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        candidates = [key]
        current = key
        for _ in range(4):
            for prefix in ("module.", "model.", "backbone.", "encoder.", "visual."):
                if current.startswith(prefix):
                    current = current[len(prefix):]
                    candidates.append(current)
        for candidate in candidates:
            if candidate in target and target[candidate].shape == value.shape:
                matched[candidate] = value
                break
    coverage = len(matched) / max(len(target), 1)
    if coverage < min_coverage:
        raise RuntimeError(f"Only {coverage:.1%} of encoder keys matched {path}; check architecture/checkpoint pairing")
    encoder.load_state_dict(matched, strict=False)


class RETFoundMAE(FoundationClassifier):
    """RETFound-MAE (ViT-L/16, 1024 features) with task-specific heads."""
    def __init__(self, num_classes: int = 39, projection_dim: int = 128,
                 checkpoint_path: str | None = "weights/RETFound_mae_shanghai.pth",
                 model_name: str = "vit_large_patch16_224", image_size: int = 224):
        encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=image_size)
        if checkpoint_path:
            _load_encoder_checkpoint(encoder, checkpoint_path)
        super().__init__(encoder, int(encoder.num_features), num_classes, projection_dim)
        self.image_size = image_size


class DINOv2Classifier(FoundationClassifier):
    """Vanilla Meta DINOv2 encoder with a linear classifier and projection head."""
    def __init__(self, num_classes: int = 39, projection_dim: int = 128,
                 model_name: str = "dinov2_vitl14", pretrained: bool = True,
                 checkpoint_path: str | None = None):
        encoder = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        if checkpoint_path:
            _load_encoder_checkpoint(encoder, checkpoint_path)
        feature_dim = int(getattr(encoder, "embed_dim", getattr(encoder, "num_features", 0)))
        if not feature_dim:
            raise RuntimeError("Could not determine DINOv2 embedding dimension")
        # Official DINOv2 hub forward returns the class-token feature vector.
        super().__init__(_FeatureBackbone(encoder, feature_dim), feature_dim, num_classes, projection_dim)


class EyeCLIPClassifier(FoundationClassifier):
    """EyeCLIP ViT-B/32 visual encoder; requires the official OpenAI CLIP package."""
    def __init__(self, num_classes: int = 39, projection_dim: int = 128,
                 checkpoint_path: str = "weights/eyeclip_visual.pt", image_size: int = 224):
        try:
            import clip
        except ImportError as exc:
            raise ImportError("EyeCLIP requires the official OpenAI CLIP package (pip install git+https://github.com/openai/CLIP.git)") from exc
        clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        _load_encoder_checkpoint(clip_model, checkpoint_path)
        # CLIP's learned visual projection produces its documented 512-d embedding.
        class VisualEncoder(nn.Module):
            def __init__(self, visual):
                super().__init__()
                self.visual = visual
                self.transformer = visual.transformer
                self.ln_post = visual.ln_post
            def forward(self, images):
                return self.visual(images)
        visual = VisualEncoder(clip_model.visual)
        super().__init__(visual, 512, num_classes, projection_dim)
        self.image_size = image_size


class BioMedCLIPClassifier(FoundationClassifier):
    """BiomedCLIP ViT-B/16 image encoder; requires open_clip_torch."""
    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, num_classes: int = 39, projection_dim: int = 128,
                 checkpoint_path: str | None = None, pretrained: bool = True):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("BiomedCLIP requires open_clip_torch (pip install open_clip_torch)") from exc
        if pretrained:
            model, _, _ = open_clip.create_model_and_transforms(self.MODEL_ID)
        else:
            model = open_clip.create_model("ViT-B-16", pretrained=None)
        if checkpoint_path:
            _load_encoder_checkpoint(model, checkpoint_path)
        # encode_image returns the official shared CLIP embedding (512 dimensions).
        class ImageEncoder(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
                self.visual = clip_model.visual
                self.transformer = getattr(self.visual, "transformer", None)
            def forward(self, images):
                return self.clip_model.encode_image(images)
        super().__init__(ImageEncoder(model), 512, num_classes, projection_dim)
        self.image_size = 224


class RETFoundDINO(FoundationClassifier):
    """RETFound-DINOv2 (ViT-L/14, 1024 features) using the shared model API."""

    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2",
        num_classes: int = 39,
        projection_dim: int = 128,
        checkpoint_path: str | None = "weights/RETFound_dinov2_shanghai.pth",
        image_size: int = 518,
    ) -> None:
        encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
        )
        if checkpoint_path:
            self._load_ret_found_checkpoint(encoder, checkpoint_path)
        super().__init__(encoder, int(encoder.num_features), num_classes, projection_dim)
        self.image_size = image_size

    @staticmethod
    def _load_ret_found_checkpoint(encoder: nn.Module, checkpoint_path: str) -> None:
        """Load RETFound's teacher/model/state_dict or raw checkpoint format."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Checkpoint at {checkpoint_path} is not a state dict")
        state = next(
            (raw[key] for key in ("teacher", "model", "state_dict") if isinstance(raw.get(key), dict)),
            raw,
        )
        target = encoder.state_dict()
        matched = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            candidates = [key]
            current = key
            for _ in range(4):
                for prefix in ("module.", "backbone.", "model.", "encoder."):
                    if current.startswith(prefix):
                        current = current[len(prefix):]
                        candidates.append(current)
            for candidate in candidates:
                if candidate in target and value.shape == target[candidate].shape:
                    matched[candidate] = value
                    break
        coverage = len(matched) / max(len(target), 1)
        if coverage < _MIN_COVERAGE_RATIO:
            raise RuntimeError(
                f"RETFound-DINO checkpoint coverage is only {coverage:.1%} "
                f"({len(matched)}/{len(target)} encoder keys matched). "
                "Check that this is a compatible RETFound-DINOv2 checkpoint."
            )
        encoder.load_state_dict(matched, strict=False)
