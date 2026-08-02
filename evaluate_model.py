import csv
import json
import math
from pathlib import Path

import hdbscan
import numpy as np
import torch
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm

from retcsl.dataset import build_unlabeled_loader
from retcsl.evaluation import ModelEvaluator
from retcsl.foundation import RETFoundDINO


DATA_DIR = Path("data/CFP_BD")
CHECKPOINT = Path("weights/RETFound_dinov2_0.2_supcon_focal_AHEH.pth")
OUT_DIR = Path("results/")
FIG_DIR = OUT_DIR / "figures"
CLASS_JSON = Path("results/AHEH_0.2/logs/class_mapping_AHEH.json")

MODEL_NAME = "vit_large_patch14_dinov2"
PROJECTION_DIM = 1024
IMAGE_SIZE = 224

BATCH_SIZE = 64
NUM_WORKERS = 8
SEED = 42

KMEANS_K_VALUES = list(range(5, 15))
KMEANS_N_INIT = 10
HDBSCAN_MCS = [10, 15, 20, 30]
HDBSCAN_MS = [5, 10, 15]
KNN_K_VALUES = [1, 3, 5, 10, 15, 20]
KNN_TEST_SIZE = 0.3


OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader, dataset = build_unlabeled_loader(
    root=DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)
n_images = len(dataset)
all_paths = [str(p) for p in dataset.paths]

with open(CLASS_JSON, "r") as f:
    class_mapping = json.load(f)
NUM_CLASSES = len(class_mapping)


model = RETFoundDINO(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    projection_dim=PROJECTION_DIM,
    checkpoint_path=None,
    image_size=IMAGE_SIZE,
).to(DEVICE)

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
try:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
except RuntimeError as e:
    raise RuntimeError(
        "Failed to load checkpoint weights — this usually means NUM_CLASSES "
        f"(derived from {CLASS_JSON}, {NUM_CLASSES} classes) does not match "
        f"the classification head baked into {CHECKPOINT}. CHECKPOINT looks "
        "like a JSIEC-trained run while CLASS_JSON currently points at an "
        "AHEH class mapping — verify these come from the same trained run "
        "before re-running.\n"
        f"Original error: {e}"
    ) from e

if missing:
    print("WARNING: checkpoint missing keys (using freshly-initialised values): %s" % (missing,))
if unexpected:
    print("WARNING: checkpoint has unexpected keys (ignored): %s" % (unexpected,))

model.eval()

evaluator = ModelEvaluator(model=model, device=DEVICE)

embeddings, _ = evaluator.extract_embeddings(loader, desc="Embeddings")
np.save(OUT_DIR / "embeddings", embeddings)

features, _ = evaluator.extract_features(loader, desc="Features")
np.save(OUT_DIR / "features", features)


all_probs = []
with torch.no_grad():
    for imgs, _ in tqdm(loader, desc="Predicting", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            logits = model(imgs, mode="classification")
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())

class_probs      = np.concatenate(all_probs, axis=0)
predicted_labels = np.argmax(class_probs, axis=1).astype(np.int32)
confidence       = class_probs[np.arange(n_images), predicted_labels]


kmeans_rows, kmeans_best = evaluator.cluster_with_kmeans(
    embeddings=embeddings,
    k_values=KMEANS_K_VALUES,
    true_labels=None,
    n_init=KMEANS_N_INIT,
    seed=SEED,
)

best_k = int(kmeans_best["k"])
kmeans_cluster_labels = KMeans(
    n_clusters=best_k, n_init=KMEANS_N_INIT, random_state=SEED
).fit_predict(embeddings.astype(np.float64))


hdbscan_rows, hdbscan_best = evaluator.cluster_with_hdbscan(
    embeddings=embeddings,
    min_cluster_sizes=HDBSCAN_MCS,
    min_samples_values=HDBSCAN_MS,
    true_labels=None,
)

hdbscan_cluster_labels = hdbscan.HDBSCAN(
    min_cluster_size=int(hdbscan_best["min_cluster_size"]),
    min_samples=int(hdbscan_best["min_samples"]),
    metric="euclidean",
    core_dist_n_jobs=-1,
).fit_predict(embeddings.astype(np.float64))


knn_rows, knn_best = evaluator.knn_probe(
    embeddings=embeddings,
    labels=predicted_labels,
    k_values=KNN_K_VALUES,
    test_size=KNN_TEST_SIZE,
    seed=SEED,
)

print("Computing 2D UMAP projection for visualization only.")
reducer = umap.UMAP(n_components=2, random_state=SEED)
umap_2d = reducer.fit_transform(embeddings.astype(np.float64))
np.save(OUT_DIR / "umap_2d", umap_2d)

evaluator.plot_clusters(
    umap_2d,
    kmeans_cluster_labels,
    title=f"KMeans clusters (k={best_k})",
    save_path=str(FIG_DIR / "kmeans_clusters.png"),
)

evaluator.plot_clusters(
    umap_2d,
    hdbscan_cluster_labels,
    title=(
        f"HDBSCAN clusters (min_cluster_size="
        f"{hdbscan_best['min_cluster_size']}, min_samples="
        f"{hdbscan_best['min_samples']})"
    ),
    save_path=str(FIG_DIR / "hdbscan_clusters.png"),
)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

manifest_rows = [
    {
        "index":          i,
        "path":           all_paths[i],
        "predicted_class": int(predicted_labels[i]),
        "confidence":     round(float(confidence[i]), 6),
        "cluster_kmeans": int(kmeans_cluster_labels[i]),
        "cluster_hdbscan": int(hdbscan_cluster_labels[i]),
    }
    for i in range(n_images)
]

_write_csv(OUT_DIR / "manifest.csv",        manifest_rows)
_write_csv(OUT_DIR / "kmeans_sweep.csv",    kmeans_rows)
_write_csv(OUT_DIR / "hdbscan_sweep.csv",   hdbscan_rows)
_write_csv(OUT_DIR / "knn_probe_sweep.csv", knn_rows)


def _sanitise(obj):
    if isinstance(obj, dict):   return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj): return None
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.ndarray):   return _sanitise(obj.tolist())
    return obj

summary = {
    "run": {
        "checkpoint": str(CHECKPOINT),
        "data_dir":   str(DATA_DIR),
        "n_images":   n_images,
        "num_classes": NUM_CLASSES,
        "seed":        SEED,
    },
    "predictions": {
        "mean_confidence": float(confidence.mean()),
        "min_confidence":  float(confidence.min()),
        "max_confidence":  float(confidence.max()),
        "class_distribution": {
            str(c): int((predicted_labels == c).sum())
            for c in sorted(np.unique(predicted_labels).tolist())
        },
    },
    "kmeans":    {"k_values_swept": KMEANS_K_VALUES, "best": kmeans_best},
    "hdbscan":   {"mcs_swept": HDBSCAN_MCS, "ms_swept": HDBSCAN_MS, "best": hdbscan_best},
    "knn_probe": {"k_values_swept": KNN_K_VALUES, "best": knn_best},
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(_sanitise(summary), f, indent=2)

print(f"Done. Results saved to {OUT_DIR}/")