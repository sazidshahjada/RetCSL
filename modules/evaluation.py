from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

import hdbscan
from sklearn.cluster import KMeans



def _cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Cluster purity: fraction of points assigned to the plurality ground-truth
    label in each cluster.  Noise points (label == -1) are excluded.

    A purity of 1.0 means every cluster is perfectly homogeneous.
    """
    mask = labels_pred >= 0
    lt = labels_true[mask]
    lp = labels_pred[mask]
    n = len(lt)
    if n == 0:
        return 0.0
    clusters: dict[int, list[int]] = {}
    for t, p in zip(lt.tolist(), lp.tolist()):
        clusters.setdefault(p, []).append(t)
    majority_sum = sum(Counter(v).most_common(1)[0][1] for v in clusters.values())
    return majority_sum / n


def _unsupervised_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute the three standard unsupervised clustering quality metrics on
    full-dimensional embeddings.  Points labelled -1 (HDBSCAN noise) are
    excluded before computation.

    Returns dict with keys: silhouette, davies_bouldin, calinski_harabasz,
    n_clusters, noise_count, noise_fraction.
    """
    mask = cluster_labels >= 0
    n_clustered = int(mask.sum())
    noise_count = int((~mask).sum())
    unique_clusters = np.unique(cluster_labels[mask])
    n_clusters = len(unique_clusters)

    result: dict[str, float] = {
        "n_clusters": float(n_clusters),
        "noise_count": float(noise_count),
        "noise_fraction": noise_count / max(len(cluster_labels), 1),
    }

    if n_clusters < 2 or n_clustered < n_clusters + 1:
        print("WARNING: Too few clustered points (%d in %d clusters) for intrinsic metrics — returning NaN." % (n_clustered, n_clusters))
        result.update({"silhouette": float("nan"),
                       "davies_bouldin": float("nan"),
                       "calinski_harabasz": float("nan")})
        return result

    X = embeddings[mask].astype(np.float64)
    y = cluster_labels[mask]

    result["silhouette"] = float(silhouette_score(X, y, metric="euclidean"))
    result["davies_bouldin"] = float(davies_bouldin_score(X, y))
    result["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
    return result


def _supervised_cluster_metrics(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute ARI, NMI, and purity against ground-truth labels.
    Noise points are excluded from ARI/NMI but purity already handles them.
    """
    mask = cluster_labels >= 0
    return {
        "ari": float(adjusted_rand_score(true_labels[mask], cluster_labels[mask])),
        "nmi": float(normalized_mutual_info_score(true_labels[mask], cluster_labels[mask])),
        "purity": _cluster_purity(true_labels, cluster_labels),
    }


def _expected_calibration_error(
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE): bins predictions by confidence (max
    softmax probability) into n_bins equal-width bins, and computes the
    weighted average gap between each bin's mean confidence and its actual
    accuracy. Lower is better; 0.0 is perfectly calibrated.
    """
    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (
            (confidences >= lo) & (confidences <= hi)
            if i == n_bins - 1
            else (confidences >= lo) & (confidences < hi)
        )
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)



class ModelEvaluator:
    """
    Post-training analysis for RETFoundDINO models.

    Args:
        model:  A RETFoundDINO (or any nn.Module with a forward(x, mode=...)
                interface).  The evaluator never modifies the model.
        device: torch.device to run inference on.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()


    def _run_batches(
        self,
        loader,
        mode: str,
        desc: str = "Inference",
    ) -> tuple[np.ndarray, list]:
        """
        Run all batches in loader through model.forward(x, mode=mode).

        Returns
        -------
        outputs : np.ndarray  shape (N, D) — concatenated float32 tensors
        meta    : list        — flat list of labels (int) or paths (str)
                               depending on what the loader yields as its
                               second element per batch
        """
        self.model.eval()
        all_outputs: list[np.ndarray] = []
        all_meta: list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                images, meta = batch
                images = images.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    out = self.model(images, mode=mode)

                # forward() with mode="both" returns a tuple; we want the
                # second element (projections) for embedding extraction.
                if isinstance(out, tuple):
                    out = out[1]

                all_outputs.append(out.float().cpu().numpy())

                # meta can be a tensor of integer labels or a tuple/list of
                # path strings — handle both.
                if isinstance(meta, torch.Tensor):
                    all_meta.extend(meta.numpy().tolist())
                else:
                    all_meta.extend(list(meta))

        return np.concatenate(all_outputs, axis=0), all_meta


    def extract_features(
        self,
        loader,
        desc: str = "Extracting features",
    ) -> tuple[np.ndarray, list]:
        """
        Extract raw backbone features (CLS token, before projection head).

        Args:
            loader: Any DataLoader yielding (images, labels_or_paths).

        Returns:
            features : np.ndarray  shape (N, embed_dim)
            meta     : list of int labels or str paths
        """
        print("Extracting backbone features (mode='features').")
        return self._run_batches(loader, mode="features", desc=desc)

    def extract_embeddings(
        self,
        loader,
        desc: str = "Extracting embeddings",
    ) -> tuple[np.ndarray, list]:
        """
        Extract L2-normalised projection embeddings (128-dim after projection head).

        These are the embeddings used for clustering.  They are already
        L2-normalised by model.forward(mode='projection').

        Args:
            loader: Any DataLoader yielding (images, labels_or_paths).

        Returns:
            embeddings : np.ndarray  shape (N, 128)  — unit-norm vectors
            meta       : list of int labels or str paths
        """
        print("Extracting L2-normalised projection embeddings (mode='projection').")
        embeddings, meta = self._run_batches(loader, mode="projection", desc=desc)

        # Verify unit norm — warn if the model's F.normalize was somehow bypassed.
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print("WARNING: Embeddings are not unit-norm (mean norm=%.4f). Check that model.forward(mode='projection') applies F.normalize." % (norms.mean(),))
        return embeddings, meta

    def predict_classes(
        self,
        loader,
        desc: str = "Predicting classes",
        use_temperature: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the classification head and return predictions, true labels, and
        class probabilities.

        Args:
            loader: DataLoader yielding (images, integer_labels).  Labels are
                    used as ground truth — this method does NOT work with
                    unlabeled loaders (use extract_embeddings instead).
            use_temperature: If True (default), logits are scaled by the
                    model's fitted temperature (see RETFoundDINO.forward /
                    FoundationModelTrainer.tune_temperature) before softmax.
                    Set False for raw, uncalibrated probabilities — useful
                    for comparing calibration before vs. after tuning.
                    Predicted classes (argmax) are identical either way,
                    since dividing by a positive scalar never changes the
                    ranking; only the reported probabilities change.

        Returns:
            predicted : np.ndarray int32  shape (N,)
            true      : np.ndarray int32  shape (N,)
            probs     : np.ndarray float32 shape (N, num_classes)
        """
        print("Running classification inference.")
        self.model.eval()

        all_preds: list[int] = []
        all_true: list[int] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16
                ):
                    logits = self.model(
                        images, mode="classification", use_temperature=use_temperature
                    )

                probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds.tolist())
                all_probs.append(probs)

                if isinstance(labels, torch.Tensor):
                    all_true.extend(labels.numpy().tolist())
                else:
                    all_true.extend(list(labels))

        return (
            np.array(all_preds, dtype=np.int32),
            np.array(all_true, dtype=np.int32),
            np.concatenate(all_probs, axis=0),
        )


    def evaluate_embedding_quality(
        self,
        embeddings: np.ndarray,
        labels: Sequence[int] | np.ndarray,
        test_size: float = 0.3,
        seed: int = 42,
        max_iter: int = 1000,
    ) -> dict[str, float]:
        """
        Measure how linearly separable the embedding space is using a fast
        logistic regression probe.  This is a standard way to quantify
        representation quality without running a full fine-tuning.

        A higher linear-probe accuracy means the embedding space organises
        classes more cleanly — directly relevant to whether clustering will
        recover disease labels.

        Args:
            embeddings : np.ndarray  shape (N, D)
            labels     : integer class labels, length N
            test_size  : fraction held out for probe evaluation
            seed       : random state for the train/test split and LR solver
            max_iter   : LogisticRegression solver iteration cap

        Returns dict with keys:
            linear_probe_accuracy, linear_probe_macro_f1,
            n_train, n_test, n_classes
        """
        labels_arr = np.array(labels, dtype=np.int32)
        n_classes = len(np.unique(labels_arr))

        X_tr, X_te, y_tr, y_te = train_test_split(
            embeddings.astype(np.float64),
            labels_arr,
            test_size=test_size,
            stratify=labels_arr,
            random_state=seed,
        )

        print("Linear probe: train=%d test=%d classes=%d" % (len(X_tr), len(X_te), n_classes))

        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        acc = float(accuracy_score(y_te, preds))
        f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))

        result = {
            "linear_probe_accuracy": acc,
            "linear_probe_macro_f1": f1,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "n_classes": n_classes,
        }
        print("Linear probe → accuracy=%.4f  macro-F1=%.4f" % (acc, f1))
        return result


    def evaluate_cluster_quality(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Evaluate a pre-computed clustering assignment.

        Intrinsic metrics (no ground truth needed):
            silhouette, davies_bouldin, calinski_harabasz

        Extrinsic metrics (only if true_labels provided):
            ari, nmi, purity

        All metrics are computed on full-dimensional embeddings.
        UMAP coordinates must never be passed here.

        Args:
            embeddings    : np.ndarray  shape (N, D)
            cluster_labels: np.ndarray  shape (N,)  — -1 = noise (HDBSCAN)
            true_labels   : optional ground-truth labels for extrinsic metrics

        Returns flat dict of all computed metrics.
        """
        result = _unsupervised_metrics(embeddings, cluster_labels)

        if true_labels is not None:
            sup = _supervised_cluster_metrics(
                np.array(true_labels, dtype=np.int32), cluster_labels
            )
            result.update(sup)
            print("Cluster quality — silhouette=%.4f | ARI=%.4f | NMI=%.4f | purity=%.4f" % (
                result.get("silhouette", float("nan")), sup["ari"], sup["nmi"], sup["purity"],
            ))
        else:
            print("Cluster quality — silhouette=%.4f | DB=%.4f | CH=%.4f (no ground truth → extrinsic metrics skipped)" % (
                result.get("silhouette", float("nan")), result.get("davies_bouldin", float("nan")), result.get("calinski_harabasz", float("nan")),
            ))

        return result


    def cluster_with_kmeans(
        self,
        embeddings: np.ndarray,
        k_values: Sequence[int],
        true_labels: Optional[np.ndarray] = None,
        n_init: int = 10,
        seed: int = 42,
    ) -> tuple[list[dict], dict]:
        """
        Sweep KMeans over a range of k values and evaluate each.

        Best k is selected by silhouette score (highest = better separation).
        If true_labels are supplied, ARI/NMI/purity are also reported per k.

        Args:
            embeddings  : np.ndarray  shape (N, D)
            k_values    : list of k values to try (e.g. range(30, 50))
            true_labels : optional ground-truth for extrinsic metrics
            n_init      : KMeans restarts per k
            seed        : random state

        Returns:
            rows     : list of per-k result dicts (suitable for pd.DataFrame)
            best_row : the row with the highest silhouette score
        """
        X = embeddings.astype(np.float64)
        rows: list[dict] = []

        print("KMeans sweep over k=%s" % (list(k_values),))
        for k in tqdm(k_values, desc="KMeans sweep"):
            km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
            cluster_labels = km.fit_predict(X)

            row: dict = {"k": k}
            row.update(_unsupervised_metrics(X, cluster_labels))
            if true_labels is not None:
                row.update(
                    _supervised_cluster_metrics(
                        np.array(true_labels, dtype=np.int32), cluster_labels
                    )
                )
            rows.append(row)
            print("DEBUG: k=%d → silhouette=%.4f%s" % (
                k, row.get("silhouette", float("nan")), f"  ARI={row['ari']:.4f}" if "ari" in row else "",
            ))

        # Best by silhouette (nan-safe).
        best_row = max(
            rows,
            key=lambda r: r.get("silhouette", float("-inf"))
            if not (isinstance(r.get("silhouette"), float) and
                    np.isnan(r.get("silhouette", 0.0)))
            else float("-inf"),
        )
        print("KMeans best: k=%d  silhouette=%.4f%s" % (best_row["k"], best_row.get("silhouette", float("nan")), f"  ARI={best_row['ari']:.4f}" if "ari" in best_row else ""))
        return rows, best_row


    def cluster_with_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_sizes: Sequence[int],
        min_samples_values: Sequence[int],
        true_labels: Optional[np.ndarray] = None,
        metric: str = "euclidean",
    ) -> tuple[list[dict], dict]:
        """
        Sweep HDBSCAN over (min_cluster_size, min_samples) pairs.

        HDBSCAN produces noise points (label -1).  Noise count and fraction
        are reported per configuration.  All intrinsic metrics exclude noise
        points.  Best configuration is chosen by silhouette score (noise excluded).

        Note: HDBSCAN is run on full-dimensional embeddings — never on UMAP
        coordinates (UMAP distorts inter-cluster distances).

        Args:
            embeddings          : np.ndarray  shape (N, D)
            min_cluster_sizes   : list of min_cluster_size values to sweep
            min_samples_values  : list of min_samples values to sweep
                                  (all combinations are tried)
            true_labels         : optional ground-truth for extrinsic metrics
            metric              : distance metric for HDBSCAN

        Returns:
            rows     : list of per-configuration result dicts
            best_row : the row with the highest silhouette (among configs with
                       at least 2 clusters and noise_fraction < 0.5)
        """
        X = embeddings.astype(np.float64)
        rows: list[dict] = []
        total = len(min_cluster_sizes) * len(min_samples_values)

        print("HDBSCAN sweep: %d configurations (min_cluster_size × min_samples)" % (total,))

        with tqdm(total=total, desc="HDBSCAN sweep") as pbar:
            for mcs in min_cluster_sizes:
                for ms in min_samples_values:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric=metric,
                        core_dist_n_jobs=-1,
                    )
                    cluster_labels = clusterer.fit_predict(X)

                    row: dict = {"min_cluster_size": mcs, "min_samples": ms}
                    row.update(_unsupervised_metrics(X, cluster_labels))
                    if true_labels is not None:
                        row.update(
                            _supervised_cluster_metrics(
                                np.array(true_labels, dtype=np.int32),
                                cluster_labels,
                            )
                        )
                    rows.append(row)
                    pbar.update(1)
                    print("DEBUG: mcs=%d ms=%d → clusters=%d noise=%.1f%%  silhouette=%.4f" % (
                        mcs, ms, int(row.get("n_clusters", 0)), row.get("noise_fraction", 0.0) * 100, row.get("silhouette", float("nan")),
                    ))

        # Best: highest silhouette among runs with ≥2 clusters and <50% noise.
        valid_rows = [
            r for r in rows
            if int(r.get("n_clusters", 0)) >= 2
            and r.get("noise_fraction", 1.0) < 0.5
            and not np.isnan(r.get("silhouette", float("nan")))
        ]
        if not valid_rows:
            print("WARNING: No HDBSCAN configuration produced ≥2 clusters with <50% noise. Returning config with most clusters as fallback.")
            best_row = max(rows, key=lambda r: int(r.get("n_clusters", 0)))
        else:
            best_row = max(valid_rows, key=lambda r: r["silhouette"])

        print("HDBSCAN best: mcs=%d ms=%d  clusters=%d  noise=%.1f%%  silhouette=%.4f%s" % (
            best_row["min_cluster_size"], best_row["min_samples"], int(best_row.get("n_clusters", 0)), best_row.get("noise_fraction", 0.0) * 100, best_row.get("silhouette", float("nan")), f"  ARI={best_row['ari']:.4f}" if "ari" in best_row else "",
        ))
        return rows, best_row


    def knn_probe(
        self,
        embeddings: np.ndarray,
        labels: Sequence[int] | np.ndarray,
        k_values: Sequence[int] = (1, 3, 5, 10, 15, 20),
        test_size: float = 0.3,
        seed: int = 42,
    ) -> tuple[list[dict], dict]:
        """
        Sweep k-Nearest-Neighbour classification over a range of k values.

        Uses cosine distance (metric="cosine", algorithm="brute"), which is
        equivalent to dot-product distance on L2-normalised embeddings and
        is the natural metric for contrastive/projection spaces.

        KNN probing is a standard lightweight alternative to linear probing:
        it requires no training, has no hyperparameters beyond k, and directly
        measures how well the embedding space clusters by class in a local
        neighbourhood sense.

        Args:
            embeddings : np.ndarray  shape (N, D)  — should be L2-normalised
            labels     : integer class labels, length N
            k_values   : k values to sweep
            test_size  : fraction held out for evaluation
            seed       : random state for the train/test split

        Returns:
            rows     : list of per-k result dicts
            best_row : row with the highest macro-F1
        """
        labels_arr = np.array(labels, dtype=np.int32)
        X = embeddings.astype(np.float64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels_arr,
            test_size=test_size,
            stratify=labels_arr,
            random_state=seed,
        )

        print("KNN probe: train=%d  test=%d  k_values=%s" % (len(X_tr), len(X_te), list(k_values)))

        rows: list[dict] = []
        for k in k_values:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric="cosine",
                algorithm="brute",   # required for cosine metric
                n_jobs=-1,
            )
            knn.fit(X_tr, y_tr)
            preds = knn.predict(X_te)

            acc = float(accuracy_score(y_te, preds))
            mac_f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
            mic_f1 = float(f1_score(y_te, preds, average="micro", zero_division=0))

            row = {"k": k, "accuracy": acc, "macro_f1": mac_f1, "micro_f1": mic_f1}
            rows.append(row)
            print("KNN k=%2d → acc=%.4f  macro-F1=%.4f  micro-F1=%.4f" % (k, acc, mac_f1, mic_f1))

        best_row = max(rows, key=lambda r: r["macro_f1"])
        print("KNN best: k=%d  macro-F1=%.4f  accuracy=%.4f" % (best_row["k"], best_row["macro_f1"], best_row["accuracy"]))
        return rows, best_row


    def evaluate_classification(
        self,
        loader,
        class_names: Optional[list[str]] = None,
        desc: str = "Evaluating test set",
        use_temperature: bool = True,
        n_bins: int = 15,
    ) -> dict:
        """
        Full classification evaluation on a labeled DataLoader.

        Runs the classification head, then computes:
            - Accuracy
            - Macro / weighted precision, recall, F1
            - Per-class precision, recall, F1, support
            - Confusion matrix
            - Expected Calibration Error (ECE)
            - sklearn classification_report string

        Args:
            loader          : DataLoader yielding (images, integer_labels)
            class_names     : optional list mapping integer index → class name string
            desc            : tqdm description
            use_temperature : If True (default), probabilities (and therefore
                              ECE) reflect the model's fitted temperature.
                              Accuracy/F1/confusion matrix are unaffected
                              either way, since argmax is invariant to
                              scaling by a positive scalar.
            n_bins          : number of bins for the reported ECE

        Returns dict with keys:
            accuracy, macro_precision, macro_recall, macro_f1,
            weighted_f1, ece, temperature, per_class (list of dicts),
            confusion_matrix (np.ndarray), classification_report (str),
            n_samples, n_classes
        """
        predicted, true, probs = self.predict_classes(
            loader, desc=desc, use_temperature=use_temperature
        )

        acc = float(accuracy_score(true, predicted))
        mac_p = float(precision_score(true, predicted, average="macro", zero_division=0))
        mac_r = float(recall_score(true, predicted, average="macro", zero_division=0))
        mac_f1 = float(f1_score(true, predicted, average="macro", zero_division=0))
        wgt_f1 = float(f1_score(true, predicted, average="weighted", zero_division=0))
        ece = _expected_calibration_error(probs, predicted, true, n_bins=n_bins)

        cm = confusion_matrix(true, predicted)
        report_str = classification_report(
            true, predicted,
            target_names=class_names,
            zero_division=0,
        )

        # Per-class breakdown as a list of dicts for easy DataFrame construction.
        classes = class_names or [str(i) for i in range(len(np.unique(true)))]
        per_class_p = precision_score(true, predicted, average=None, zero_division=0)
        per_class_r = recall_score(true, predicted, average=None, zero_division=0)
        per_class_f = f1_score(true, predicted, average=None, zero_division=0)
        support = np.bincount(true, minlength=len(classes))

        per_class = [
            {
                "class": classes[i] if i < len(classes) else str(i),
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f[i]),
                "support": int(support[i]),
            }
            for i in range(len(per_class_p))
        ]

        result = {
            "accuracy": acc,
            "macro_precision": mac_p,
            "macro_recall": mac_r,
            "macro_f1": mac_f1,
            "weighted_f1": wgt_f1,
            "ece": ece,
            "temperature": float(self.model.temperature.item()) if hasattr(self.model, "temperature") else None,
            "per_class": per_class,
            "confusion_matrix": cm,
            "classification_report": report_str,
            "n_samples": len(true),
            "n_classes": len(np.unique(true)),
        }

        print(
            "Classification eval → accuracy=%.4f  macro-F1=%.4f  weighted-F1=%.4f  ECE=%.4f  n=%d  classes=%d"
            % (acc, mac_f1, wgt_f1, ece, len(true), len(np.unique(true)))
        )
        print("\n%s" % (report_str,))

        return result


    def evaluate_calibration(
        self,
        loader,
        n_bins: int = 15,
        desc: str = "Evaluating calibration",
    ) -> dict[str, float]:
        """
        Compare Expected Calibration Error (ECE) before vs. after temperature
        scaling, on the same loader. This is the direct check that
        FoundationModelTrainer.tune_temperature() actually improved
        calibration rather than just changing numbers.

        If tune_temperature() hasn't been run yet, "after" will equal
        "before" since the model's temperature is still 1.0 (a no-op).

        Args:
            loader : DataLoader yielding (images, integer_labels)
            n_bins : number of equal-width confidence bins for ECE

        Returns dict with keys:
            ece_before_scaling, ece_after_scaling, temperature, accuracy
            (accuracy is identical regardless of temperature — argmax is
            invariant to positive scalar scaling).
        """
        preds_raw, true, probs_raw = self.predict_classes(
            loader, desc=f"{desc} (raw)", use_temperature=False
        )
        preds_scaled, _, probs_scaled = self.predict_classes(
            loader, desc=f"{desc} (scaled)", use_temperature=True
        )

        ece_before = _expected_calibration_error(probs_raw, preds_raw, true, n_bins=n_bins)
        ece_after = _expected_calibration_error(probs_scaled, preds_scaled, true, n_bins=n_bins)
        acc = float(accuracy_score(true, preds_scaled))
        temperature = float(self.model.temperature.item())

        result = {
            "ece_before_scaling": ece_before,
            "ece_after_scaling": ece_after,
            "temperature": temperature,
            "accuracy": acc,
        }
        print(
            "Calibration — T=%.4f | ECE before=%.4f after=%.4f | accuracy=%.4f (unaffected by T)"
            % (temperature, ece_before, ece_after, acc)
        )
        return result


    @staticmethod
    def _scatter_panel(
        ax,
        coords: np.ndarray,
        labels: np.ndarray,
        panel_title: str,
        noise_label: Optional[int] = None,
        class_names: Optional[list[str]] = None,
        point_size: float = 8.0,
    ) -> None:
        """
        Draw one colored scatter panel. Points equal to noise_label (if given)
        are drawn as light gray 'x' markers rather than a normal color, so
        HDBSCAN noise is visually distinct from a real cluster.
        """
        labels_arr = np.asarray(labels)
        unique = sorted(set(labels_arr.tolist()))
        cmap = plt.get_cmap("tab20")

        for i, lab in enumerate(unique):
            mask = labels_arr == lab
            if noise_label is not None and lab == noise_label:
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=point_size, c="lightgray", marker="x",
                    alpha=0.5, label="noise",
                )
                continue
            color = cmap(i % 20)  # cycles if >20 groups — see plot_clusters docstring
            name = (
                class_names[lab]
                if class_names is not None and 0 <= lab < len(class_names)
                else str(lab)
            )
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=point_size, color=color, alpha=0.7, label=name,
            )

        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(unique) <= 20:
            ax.legend(markerscale=2, fontsize=7, loc="best", ncol=1 if len(unique) <= 10 else 2)


    def plot_clusters(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        method: str = "umap",
        class_names: Optional[list[str]] = None,
        seed: int = 42,
        point_size: float = 8.0,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        reducer_kwargs: Optional[dict] = None,
    ):
        """
        Scatter-plot embeddings in 2D, colored by predicted cluster, with an
        optional side-by-side panel colored by ground-truth label so cluster
        quality is visually inspectable (per spec §10).

        Visualization only: reduction to 2D is never used for clustering
        itself — KMeans/HDBSCAN/GMM must always run on the full-dimensional
        embeddings passed to cluster_with_kmeans / cluster_with_hdbscan.

        Args:
            embeddings     : np.ndarray shape (N, D). If D == 2, treated as an
                             already-computed 2D projection (e.g. a Stage-2
                             umap_2d.npy) and used as-is — no re-reduction.
            cluster_labels : predicted cluster id per point; -1 = HDBSCAN noise.
            true_labels    : optional ground-truth label per point. If given,
                             a second panel is added next to the cluster panel.
            method         : "umap" or "tsne" — only used when embeddings are
                             not already 2D.
            class_names    : optional list mapping int true_label -> name,
                             used for the ground-truth panel's legend.
            seed           : random_state for the reducer.
            point_size     : matplotlib scatter marker size.
            title          : optional figure-level title.
            save_path      : if given, saves the figure here (dpi=150) and
                             closes it, returning None. Otherwise the Figure
                             is returned for further use (e.g. in a notebook).
            reducer_kwargs : extra kwargs forwarded to the UMAP/TSNE constructor.

        Returns:
            matplotlib.figure.Figure, or None if save_path was given.

        Note: color cycling uses a 20-color colormap; with >20 distinct
        clusters/labels, colors repeat and the legend is suppressed for
        readability — inspect the returned figure closely in that case.
        """
        reducer_kwargs = reducer_kwargs or {}

        if embeddings.shape[1] == 2:
            coords = embeddings
        else:
            print(
                "Reducing %d-dim embeddings to 2D with %s for visualization only."
                % (embeddings.shape[1], method)
            )
            if method == "umap":
                import umap
                reducer = umap.UMAP(n_components=2, random_state=seed, **reducer_kwargs)
            elif method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=seed, **reducer_kwargs)
            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'umap' or 'tsne'.")
            coords = reducer.fit_transform(embeddings.astype(np.float64))

        n_panels = 2 if true_labels is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        axes = [axes] if n_panels == 1 else list(axes)

        self._scatter_panel(
            axes[0], coords, np.asarray(cluster_labels), "Predicted clusters",
            noise_label=-1, point_size=point_size,
        )

        if true_labels is not None:
            self._scatter_panel(
                axes[1], coords, np.asarray(true_labels), "Ground-truth labels",
                class_names=class_names, point_size=point_size,
            )

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print("Cluster plot saved to %s" % (save_path,))
            return None

        return fig