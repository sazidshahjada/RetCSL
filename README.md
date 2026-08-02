# **RetCSL: Contrastive-Balanced-Focal Fine- Tuning for Disease-Aware Retinal Foundation Model Representations**

## **1. Abstract**

Foundation models pretrained on large collections of color fundus photographs (CFP) — such as RETFound, EyeCLIP, and FLAIR — provide strong general-purpose retinal representations, but their pretraining objectives (masked reconstruction, self-distillation, or generic image-text contrast) do not explicitly encode disease-discriminative structure. As a result, the embedding spaces of these models are not reliably organized around clinically meaningful disease boundaries, and this weakness is most pronounced for rare conditions that are underrepresented in any single training corpus. This project proposes **RetCSL**, a fine-tuning framework that adapts existing retinal foundation models into disease-aware feature extractors using a combined **Class-Balanced Focal Loss** and **Supervised Contrastive Loss**, trained on a harmonized union of open multi-disease CFP datasets. The two loss terms are complementary by design: the balanced focal term corrects classification-boundary bias toward common diseases, while the contrastive term builds embedding-space structure from pairwise relationships that remain informative even for classes with only a handful of examples. Only the last 4–5 transformer blocks of each frozen backbone are fine-tuned, keeping the approach computationally tractable on a single consumer GPU (RTX 3060, 12 GB). The project's central empirical claim — that this combined objective specifically improves rare-disease representation and recognition, not just aggregate accuracy — will be evaluated through a dedicated frequency-stratified ablation study across multiple foundation-model backbones and multiple open datasets.

## **2. Background and Motivation**

Retinal fundus photography remains one of the most cost-effective screening modalities for detecting both ocular and systemic disease, which is especially relevant in low-resource healthcare settings such as Bangladesh and much of South Asia, where ophthalmologist-to-patient ratios are low and diabetic retinopathy, glaucoma, and other blinding conditions are frequently diagnosed late. Recent retinal foundation models have substantially advanced automated screening by providing transferable representations learned from large amounts of unlabeled or weakly labeled data.

However, prior work on frozen-encoder pipelines for retinal disease clustering and pseudo-labeling (the author's own preceding project, RetCluster) surfaced a specific limitation: a frozen foundation-model embedding space, however strong on generic image similarity, is not guaranteed to separate diseases cleanly — and published evaluations support this concern. In particular, comparative evaluation of the FLAIR foundation model against RETFound found that despite substantially less pretraining data and a smaller backbone, FLAIR outperformed RETFound under linear probing, with the authors explicitly noting that self-supervised retinal encoders typically require deeper fine-tuning, not just frozen feature extraction, to adapt well to a target task. This motivates a shift from frozen-embedding approaches toward targeted, resource-efficient fine-tuning — the approach this project takes.

A second, compounding challenge is that any realistic multi-disease retinal dataset is severely long-tailed: a small number of common conditions (diabetic retinopathy, cataract, glaucoma) dominate sample counts, while dozens of clinically significant but rarer conditions are represented by only a few dozen images each, even after combining multiple public sources. Standard supervised fine-tuning, even with reweighted losses, tends to still under-serve rare classes, because a classification decision boundary for a rare class receives comparatively few informative gradient updates over training. This project's central methodological bet is that combining a decision-boundary-level correction (Class-Balanced Focal Loss) with a representation-geometry-level correction (Supervised Contrastive Loss) addresses this problem from two complementary angles simultaneously.

## **3. Problem Statement**

Existing retinal foundation models produce embeddings that are not explicitly optimized for disease discriminability, and standard supervised fine-tuning on long-tailed multi-disease data tends to under-represent rare disease classes in both classification performance and embedding-space structure. There is currently no systematic study evaluating whether a combined contrastive-plus-balanced-focal fine-tuning objective — applied efficiently via partial (last-block) fine-tuning — measurably improves rare-disease representation quality across multiple retinal foundation-model backbones and multiple open multi-disease datasets.

## **4. Research Objectives**

1. Design and implement a combined Class-Balanced Focal + Supervised Contrastive loss for multi-label, long-tailed retinal disease classification, including a principled definition of positive/negative pairs for the multi-label contrastive term.
2. Construct a harmonized multi-disease CFP training corpus by unifying the label taxonomies of several open datasets (RFMiD 1.0/2.0, ODIR-5K, JSIEC-1000, BRSET), using a masked multi-label annotation scheme that avoids penalizing a model for labels a given source dataset never assessed.
3. Fine-tune the last 4–5 transformer blocks of multiple retinal and general-purpose foundation-model backbones (RETFound-MAE, RETFound-DINO, EyeCLIP, FLAIR, BiomedCLIP, DINOv2) under this objective, within a 12 GB consumer-GPU compute budget.
4. Empirically demonstrate, via a frequency-stratified (many/medium/few-shot class) evaluation protocol, that the combined objective improves rare-disease classification and embedding-space quality specifically, relative to a balanced-focal-only baseline.
5. Characterize the method's sensitivity to key design choices (loss weighting, fine-tuning depth, batch composition/negative pool size, dataset composition) via a structured ablation study suitable for a MICCAI/MIDL-style submission.

## **5. Related Work (Summary)**

* **Retinal foundation models.** RETFound (masked-autoencoder and DINOv2-based variants) is pretrained on approximately 1.6 million retinal images and has become a widely used backbone for downstream retinal tasks. EyeCLIP extends CLIP-style contrastive pretraining across 11 ophthalmic imaging modalities. FLAIR incorporates expert clinical knowledge as text supervision in a contrastive vision-language framework and has shown strong transfer performance despite smaller scale. BiomedCLIP and DINOv2 serve as non-ophthalmic-specific controls (biomedical-general and natural-image-general, respectively).
* **Long-tailed and imbalanced classification.** Class-Balanced Loss reweights per-class loss contributions by the effective number of samples per class; Focal Loss down-weights well-classified (typically majority-class) examples to focus learning capacity on harder examples. This project combines both principles in a multi-label setting.
* **Supervised contrastive learning.** Supervised Contrastive Loss extends contrastive representation learning by using label information to define positive pairs, producing embeddings with improved inter-class separation compared to purely self-supervised or purely cross-entropy-trained representations. Its extension to a multi-label setting (where "positive" must be defined via label-set overlap rather than label equality) is a specific methodological contribution of this project.
* **Multi-dataset harmonization.** Prior work assembling multiple public fundus datasets into unified benchmarks (e.g., the MultiEYE benchmark, which aggregates 13 public fundus datasets alongside OCT data) demonstrates both the value and the annotation-consistency challenges of this approach, and directly informs the harmonization protocol proposed here.

## **6. Proposed Methodology**

### **6.1 Datasets and Harmonization**

The primary training corpus combines four open multi-label, multi-disease CFP datasets: **RFMiD 1.0** (~3,200 images, 45–46 disease/lesion categories, severe long tail), **RFMiD 2.0** (~860 additional images extending the same taxonomy), **ODIR-5K** (~8,000 images, 8 coarse multi-label disease categories), and **BRSET** (16,266 images, Brazilian population, multi-label disease and demographic annotations). **JSIEC-1000** (1,000 images, 39 fine-grained categories) serves as an additional diversity/generalization source. A unified multi-label taxonomy is constructed by manually cross-walking each dataset's label vocabulary, and — critically — each image is annotated with both a multi-hot label vector *and* an annotation-mask vector recording which taxonomy entries its source dataset actually assessed, so that unassessed labels contribute no loss signal rather than being treated as confirmed negatives. Optional secondary DR-severity datasets (APTOS 2019, IDRiD) may be incorporated via a separate ordinal severity head, kept architecturally distinct from the primary multi-label disease head.

### **6.2 Model Architecture**

Each foundation-model backbone is used with its first N transformer blocks frozen (forward pass under no\_grad(), no gradient computation or activation retention) and its last 4–5 blocks fine-tuned. The backbone feature vector feeds two heads: (i) a multi-label classification head (per-class sigmoid outputs over the unified taxonomy) and (ii) a projection head (reducing to a 128-dimensional, L2-normalized embedding) used exclusively for the contrastive loss term.

### **6.3 Combined Loss Function**

Training minimizes L = L\_balanced\_focal + λ · L\_contrastive, where L\_balanced\_focal is a mask-aware, per-class Class-Balanced Focal Loss applied to the multi-label classification head, and L\_contrastive is a multi-label-adapted Supervised Contrastive Loss applied to the projection head, with positive pairs defined by shared-label overlap between images in a batch. The weighting term λ is tuned as part of the ablation study (Section 6.6).

### **6.4 Training Strategy**

Batches are constructed via a multi-label-aware class-balanced sampler ensuring adequate representation of rare classes per batch (a requirement for the contrastive term to receive valid positive-pair signal for those classes). Training uses AdamW with a cosine learning-rate schedule, bf16 mixed precision, gradient accumulation, and gradient clipping, scoped to fit a 12 GB VRAM budget (RTX 3060), with Google Colab Pro (single A100) available for select larger-scale ablation configurations.

### **6.5 Evaluation Protocol**

Evaluation reports standard multi-label metrics (per-label precision/recall/F1, macro/micro F1, mean Average Precision) alongside embedding-quality metrics (Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, k-NN accuracy). Critically, **all metrics are reported separately for many-shot, medium-shot, and few-shot (rare) class groups**, since the project's central claim concerns the rare-class group specifically, not aggregate performance.

### **6.6 Ablation Studies**

Planned ablations include: (A) loss-component isolation (focal-only vs. contrastive-only vs. combined); (B) λ sensitivity sweep; (C) fine-tuning depth (frozen through full fine-tune); (D) contrastive negative-pool size and an optional memory-queue mechanism for small-batch settings; (E) **the central rare-class-focus ablation**, comparing per-frequency-group performance between the balanced-focal-only baseline and the combined objective; and (F) dataset-composition ablation (single-source vs. combined training). Ablations A–C and E–F are run in full on one anchor backbone (RETFound-MAE); the remaining backbones are evaluated once at the best configuration to produce a generalization table.

## **7. Expected Contributions**

1. A combined Contrastive + Class-Balanced Focal fine-tuning framework, specifically adapted to the multi-label setting, for building disease-aware representations from retinal foundation models.
2. A masked multi-label annotation scheme enabling principled harmonization of multiple open retinal datasets with incompatible label taxonomies, without introducing false-negative label noise.
3. Empirical evidence, evaluated across multiple foundation-model backbones, on whether and to what extent this approach specifically improves rare-disease recognition and representation — a question not directly addressed by prior single-backbone or single-dataset retinal fine-tuning studies.
4. A resource-efficient (last-block, single consumer-GPU) fine-tuning recipe that lowers the barrier for similar work in compute-constrained research settings, relevant to the project's broader goal of equitable clinical AI research capacity in South Asia.

## **8. Work Plan (3-Month Timeline)**

This is a deliberately compressed, general-purpose 3-month (12-week) workplan. To fit this window, the full six-backbone generalization sweep and the full supplementary ablation set from the extended project design are trimmed down; what's deferred is noted explicitly below rather than silently dropped.

| **Week(s)** | **Focus** | **Key deliverables** |
| --- | --- | --- |
| 1 | Environment setup, dataset acquisition | Working environment for the anchor backbone; RFMiD, ODIR-5K, JSIEC, BRSET downloaded and inspected |
| 2 | Label harmonization | Unified multi-label taxonomy, crosswalk table, annotation-mask generation logic |
| 3 | Data pipeline | Dataset loaders, augmentation pipeline, class-balanced/P-K sampler, validated with a throwaway batch-inspection script |
| 4 | Model + loss implementation | Backbone loader with partial-freeze logic (anchor backbone), multi-label classifier + projection heads, mask-aware Balanced Focal + multi-label Contrastive loss, unit-validated on a tiny batch |
| 5–6 | End-to-end training, anchor backbone | Balanced-Focal-only baseline trained and validated; combined-loss (Focal+Contrastive) run trained and compared against baseline |
| 7 | Embedding + rare-class evaluation tooling | Frequency-stratified (many/medium/few-shot) metrics, embedding-quality metrics, UMAP visualization pipeline |
| 8–9 | Core ablation suite (anchor backbone) | Loss-component ablation (A), lambda sensitivity (B), fine-tuning-depth ablation (C), and the central rare-class-focus ablation (E) |
| 10 | Limited generalization check | Best configuration re-run on 1–2 additional backbones (e.g., EyeCLIP and/or FLAIR) to test whether findings hold beyond the anchor backbone |
| 11 | Final test-set evaluation | Classification report, confusion matrix, ROC/PR curves, embedding visualization, and failure-case analysis on the held-out test set |
| 12 | Consolidation and write-up | Results tables/figures finalized; internal draft of the method and results sections |

**Explicitly deferred beyond this 3-month window** (candidates for a follow-on phase once initial results are in hand): the full six-backbone generalization table, the dataset-composition ablation (F), the negative-pool/memory-queue ablation (D), and the optional clustering-bridge experiment connecting RetCSL back to RetCluster.

## **9. Resource Requirements**

* **Compute:** primary development on a local RTX 3060 (12 GB VRAM); secondary access to Google Colab Pro (single A100) for select larger-scale ablation runs.
* **Data:** RFMiD 1.0/2.0, ODIR-5K, JSIEC-1000, BRSET (all publicly available; access/license terms to be confirmed per dataset before any redistribution of derived harmonized-label files); optionally APTOS 2019/IDRiD for a secondary severity-grading task.
* **Software:** PyTorch, timm, backbone-specific repositories (RETFound, EyeCLIP, FLAIR, BiomedCLIP, DINOv2), scikit-learn, umap-learn, iterative-stratification (multi-label stratified splitting), standard experiment-tracking/logging tools.
* **Personnel:** primary researcher (author); potential supervisory/collaborative input welcomed on experimental design and manuscript positioning.

## **10. Risk Assessment and Mitigation**

| **Risk** | **Mitigation** |
| --- | --- |
| Label harmonization introduces systematic annotation noise across datasets | Explicit annotation-mask mechanism (Section 6.1); manual review of the taxonomy crosswalk; per-source performance reporting to detect source-specific anomalies |
| 12 GB VRAM insufficient for target batch sizes needed by contrastive learning | Partial (last-block) fine-tuning to reduce activation memory; gradient accumulation; optional memory-queue mechanism to decouple negative-pool size from batch size; Colab Pro/A100 as burst capacity |
| Rare-class improvement does not materialize or is not statistically distinguishable from baseline | This is itself a valid, reportable scientific outcome; the frequency-stratified evaluation protocol is designed to detect partial or null effects honestly, not only to confirm the hypothesis |
| Backbone dependency conflicts (particularly EyeCLIP's older pinned dependencies vs. newer RETFound requirements) | Isolated per-backbone environments; anchor-backbone-first development order so core pipeline is validated before adding less-compatible backbones |
| Domain shift across source datasets (different cameras/populations) confounds results | Per-source-dataset stratified splitting before combination; per-source performance breakdown reported alongside aggregate results |

## **11. Expected Outcomes and Dissemination**

The primary outcome is a submission-ready manuscript presenting the RetCSL framework, its rare-disease-focused empirical evaluation, and the multi-dataset harmonization methodology as a reusable contribution. A secondary, optional outcome is a demonstration that a RetCSL-fine-tuned encoder improves downstream clustering/pseudo-labeling quality when substituted into the author's related RetCluster pipeline on the BanglaEye dataset, which — if positive — would support a broader narrative connecting representation-quality improvement to practical label-efficient screening pipelines for South Asian clinical settings.

## **12. Indicative References**

* Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR*.
* Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
* Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. *NeurIPS*.
* Zhou, Y., et al. RETFound: a foundation model for generalizable disease detection from retinal images. *Nature*.
* Michi-3000 et al. EyeCLIP: A Vision-Language Foundation Model for Multi-Modal Ophthalmic Image Analysis.
* Silva-Rodriguez, J., Chakor, H., Kobbi, R., Dolz, J., & Ben Ayed, I. FLAIR: A Foundation Language-Image Model of the Retina.
* Pham, T. X., et al. Panwar, S., et al. RFMiD: Retinal Fundus Multi-Disease Image Dataset.
* Li, N., Li, T., Hu, C., Wang, K., & Kang, H. (2021). A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-Disease Detection. *BOE*.
* Nakayama, L. F., et al. A Brazilian Multilabel Ophthalmological Dataset (BRSET).
* MultiEYE: A benchmark aggregating public fundus and OCT datasets for multi-disease recognition.