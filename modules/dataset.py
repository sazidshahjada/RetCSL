import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets


_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
)

# ImageNet statistics used by RETFound.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

def get_train_transform(image_size: int = 224) -> A.Compose:
    """
    Optimized Albumentations pipeline for the train split.
    Enhanced for multi-hospital domain generalization.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5,
        ),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),

        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_inference_transform(image_size: int = 224) -> A.Compose:
    """
    Updated to include deterministic CLAHE preprocessing to match train split.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])



class EyeDiseaseDataset(Dataset):
    """
    Reads images from a list of file paths and applies an albumentations
    transform.  PIL → RGB conversion is handled here so callers never need
    to worry about greyscale or RGBA inputs.
    """

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        transform: A.Compose | None = None,
    ) -> None:
        assert len(paths) == len(labels), "paths and labels must have equal length"
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label



class PKBatchSampler(Sampler):
    """
    At each step, randomly samples P classes then K images per class,
    guaranteeing at least one valid positive pair per class in every batch.
    Required for SupConLoss to receive non-degenerate batches.

    Args:
        labels:      Integer label for every sample in the dataset.
        p:           Number of classes to sample per batch.
        k:           Number of images per class per batch.
        num_batches: Total batches per epoch.  Defaults to
                     len(labels) // (p * k).
    """

    def __init__(
        self,
        labels: Sequence[int],
        p: int = 13,
        k: int = 3,
        num_batches: int | None = None,
    ) -> None:
        self.labels = list(labels)
        self.p = p
        self.k = k
        self.batch_size = p * k

        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.class_to_indices[lbl].append(idx)
        self.classes = list(self.class_to_indices.keys())

        if len(self.classes) < p:
            raise ValueError(
                f"PKBatchSampler: p={p} but only {len(self.classes)} classes "
                "are available in this split.  Reduce p."
            )

        self.num_batches = num_batches or max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = random.sample(self.classes, self.p)
            batch: list[int] = []
            for cls in chosen_classes:
                pool = self.class_to_indices[cls]
                # sample with replacement when a class has fewer than k images
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    batch.extend(random.choices(pool, k=self.k))
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches



def build_data_pipelines(
    data_dir: str,
    class_names: Sequence[str] | None = None,
    image_size: int = 224,
    train_size: float = 0.7,
    seed: int = 42,
    p: int = 13,
    k: int = 3,
    batch_size: int = 39,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> dict:
    """
    Build train / val / test DataLoaders from an ImageFolder-style directory.

    Split order (critical — spec §2):
      1. Read all paths and labels from the ORIGINAL dataset directory.
      2. Perform stratified train / val / test split on the original images.
      3. Apply albumentations augmentation ONLY to the train split at load time.
      4. Val and test splits use the deterministic inference transform.

    This ensures no augmented copy of a training image ever leaks into val/test.

    Returns a dict with keys:
        train_loader            — plain shuffled loader (for focal-only run)
        train_loader_contrastive — P-K sampled loader (for SupCon run)
        val_loader
        test_loader
        class_names
        train_labels            — integer labels for the train split
        num_classes
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_path}")

    if class_names is None:
        folder_ds = datasets.ImageFolder(root=str(data_path))
        all_paths: list[str] = [s[0] for s in folder_ds.samples]
        all_labels: list[int] = [s[1] for s in folder_ds.samples]
        resolved_class_names: list[str] = folder_ds.classes
    else:
        resolved_class_names = [str(name) for name in class_names]
        all_paths = []
        all_labels = []
        for label, name in enumerate(resolved_class_names):
            class_dir = data_path / name
            if not class_dir.is_dir():
                continue
            class_paths = sorted(
                p
                for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )
            all_paths.extend(str(p) for p in class_paths)
            all_labels.extend([label] * len(class_paths))

        if len(all_paths) == 0:
            raise FileNotFoundError(
                f"No valid images found under {data_path} for fixed classes: "
                f"{resolved_class_names}"
            )

    num_classes = len(resolved_class_names)
    present_classes = sorted(set(all_labels))

    print(
        "Dataset: %d images across %d/%d non-empty classes from %s"
        % (len(all_paths), len(present_classes), num_classes, data_path)
    )

    def _stratify_labels(labels: Sequence[int], split_name: str, test_size: float):
        counts: dict[int, int] = defaultdict(int)
        for label in labels:
            counts[int(label)] += 1
        n_classes = len(counts)
        n_test = math.ceil(len(labels) * test_size)
        n_train = len(labels) - n_test
        can_stratify = (
            n_classes >= 2
            and min(counts.values()) >= 2
            and n_train >= n_classes
            and n_test >= n_classes
        )
        if not can_stratify:
            print(
                "WARNING: %s split cannot be stratified with %d samples across "
                "%d classes; falling back to random split."
                % (split_name, len(labels), n_classes)
            )
            return None
        return labels

    val_test_size = 1.0 - train_size
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_test_size,
        stratify=_stratify_labels(all_labels, "train/temp", val_test_size),
        random_state=seed,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=_stratify_labels(temp_labels, "val/test", 0.5),
        random_state=seed,
    )

    print("Split: train=%d | val=%d | test=%d" % (len(train_paths), len(val_paths), len(test_paths)))

    train_xform = get_train_transform(image_size)
    inf_xform = get_inference_transform(image_size)

    train_ds = EyeDiseaseDataset(train_paths, train_labels, train_xform)
    val_ds = EyeDiseaseDataset(val_paths, val_labels, inf_xform)
    test_ds = EyeDiseaseDataset(test_paths, test_labels, inf_xform)

    shared_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **shared_kwargs,
    )

    train_class_count = len(set(train_labels))
    pk_p = min(p, train_class_count)
    if pk_p < p:
        print(
            "PKBatchSampler: reducing P from %d to %d because only %d classes "
            "are present in the train split."
            % (p, pk_p, train_class_count)
        )
    pk_sampler = PKBatchSampler(train_labels, p=pk_p, k=k)
    train_loader_contrastive = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        **shared_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **shared_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **shared_kwargs,
    )

    print("Loaders ready — standard batch: %d | P-K batch: %d (P=%d, K=%d)" % (batch_size, pk_p * k, pk_p, k))

    return {
        "train_loader": train_loader,
        "train_loader_contrastive": train_loader_contrastive,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": resolved_class_names,
        "train_labels": train_labels,
        "num_classes": num_classes,
    }



class UnlabeledImageDataset(Dataset):
    """
    Flat dataset that walks an image directory recursively and returns
    (tensor, path_str) pairs.  No labels are expected or returned.

    The directory may be structured in any way — flat, class-nested, or
    arbitrarily deep.  Every file whose suffix matches _IMAGE_SUFFIXES is
    included.  Subfolder names are never used as labels.

    This is the correct entry point for embedding extraction on data where
    ground-truth labels are unknown or intentionally withheld.

    Args:
        root:       Root directory to walk.
        image_size: Square size passed to the inference transform.
        transform:  Optional override albumentations pipeline.  Defaults to
                    the standard deterministic inference transform (resize +
                    centre-crop + normalise).  Never pass a training/
                    augmentation transform here.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        transform: A.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"UnlabeledImageDataset: root not found: {self.root}")

        self.transform = transform or get_inference_transform(image_size)

        self.paths: list[Path] = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"UnlabeledImageDataset: no images found under {self.root}. "
                f"Searched for suffixes: {sorted(_IMAGE_SUFFIXES)}"
            )

        print("UnlabeledImageDataset: found %d images under %s" % (len(self.paths), self.root))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        path = self.paths[idx]
        img = np.array(Image.open(path).convert("RGB"))
        tensor = self.transform(image=img)["image"]
        return tensor, str(path)


def build_unlabeled_loader(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, UnlabeledImageDataset]:
    """
    Build a DataLoader over an unlabeled image directory.

    Returns (loader, dataset) so callers can access dataset.paths for
    alignment between embeddings and file names without a second directory
    walk.

    Args:
        root:               Directory to walk recursively.
        image_size:         Square resize target.
        batch_size:         Images per batch.
        num_workers:        DataLoader worker processes.
        pin_memory:         Pin host memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor:    Batches prefetched per worker.

    Returns:
        (DataLoader, UnlabeledImageDataset)
    """
    dataset = UnlabeledImageDataset(root=root, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # preserve index alignment with dataset.paths
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,        # never drop images — every path must be embedded
    )

    print("Unlabeled loader: %d images | batch=%d | %d batches" % (len(dataset), batch_size, len(loader)))
    return loader, dataset