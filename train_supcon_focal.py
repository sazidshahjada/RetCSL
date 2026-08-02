import json
import os

import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from retcsl.dataset import build_data_pipelines
from retcsl.foundation import RETFoundDINO
from retcsl.losses import ClassBalancedFocalLoss, CombinedLoss, SupConLoss
from retcsl.trainer import FoundationModelTrainer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/AHEH"
AUGMENTED_DATA_DIR = "data/AHEH_Augmented"
IMAGE_SIZE = 224
MODEL_NAME = "vit_large_patch14_dinov2"
PRETRAINED_WEIGHT = "weights/RETFound_dinov2_shanghai.pth"

NUM_BLOCKS_TO_UNFREEZE = 8
PROJECTION_DIM = 1024
LAMBDA = 0.2
NUM_EPOCHS = 200
PATIENCE = 7
LR = 1e-5
WEIGHT_DECAY = 5e-2
MAX_GRAD_NORM = 1.0

P, K = 10, 4

RESULT_DIR = f"results/AHEH_{LAMBDA}"
FIG_DIR = os.path.join(RESULT_DIR, "figures")
LOG_DIR = os.path.join(RESULT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
for d in [FIG_DIR, LOG_DIR, CHECKPOINT_DIR, "weights"]:
    os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    pipelines = build_data_pipelines(
        data_dir=AUGMENTED_DATA_DIR,
        image_size=IMAGE_SIZE,
        train_size=0.7,
        seed=42,
        p=P,
        k=K,
        batch_size=P * K,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    train_loader_c = pipelines["train_loader_contrastive"]
    val_loader = pipelines["val_loader"]
    test_loader = pipelines["test_loader"]
    class_names = pipelines["class_names"]
    train_labels = pipelines["train_labels"]
    num_classes = pipelines["num_classes"]

    with open(os.path.join(LOG_DIR, "class_mapping_AHEH.json"), "w") as f:
        json.dump(class_names, f, indent=4)

    original_ds = datasets.ImageFolder(root=DATA_DIR)
    samples_per_class = [
        sum(1 for lbl in original_ds.targets if lbl == i) for i in range(num_classes)
    ]
    print("Samples per class:", samples_per_class)
    
    focal_loss = ClassBalancedFocalLoss(samples_per_class=samples_per_class)
    supcon_loss = SupConLoss(temperature=0.07)
    criterion = CombinedLoss(focal_loss, supcon_loss, lam=LAMBDA)


    model = RETFoundDINO(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        projection_dim=PROJECTION_DIM,
        checkpoint_path=PRETRAINED_WEIGHT,
        image_size=IMAGE_SIZE
    ).to(DEVICE)


    model.unfreeze_last_n_blocks(NUM_BLOCKS_TO_UNFREEZE)
    model.count_parameters()


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameter tensors: %d" % (len(trainable_params),))
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)


    trainer = FoundationModelTrainer(
        model=model,
        train_loader_contrastive=train_loader_c,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, f"best_model_supcon_focal_{LAMBDA}.pth"),
        patience=PATIENCE,
        max_grad_norm=MAX_GRAD_NORM,
    )


    print(f"Starting training — lambda={LAMBDA}\n")
    model, history = trainer.fit(num_epochs=NUM_EPOCHS)
    fitted_temperature = trainer.tune_temperature(test_loader)

    torch.save(
        model.state_dict(),
        f"weights/RETFound_dinov2_{LAMBDA}_supcon_focal_AHEH.pth",
    )
    print("Final weights saved.")


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss_total"], label="Total")
    axes[0].plot(history["train_loss_focal"], label="Focal component")
    axes[0].plot(history["train_loss_supcon"], label="SupCon component")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss curves — Focal + SupCon (λ={LAMBDA})")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val accuracy")
    axes[1].plot(history["val_f1"], label="Val macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Validation metrics — Focal + SupCon (λ={LAMBDA})")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"curves_supcon_focal_{LAMBDA}.png"), dpi=150)
    plt.close(fig)

    test_metrics = trainer.evaluate(test_loader, class_names=class_names)
    with open(os.path.join(LOG_DIR, f"test_metrics_supcon_focal_{LAMBDA}.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("Done.  Test metrics: %s" % (test_metrics))