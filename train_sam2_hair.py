import os
import cv2
import json
import math
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    Sam2Model,
    Sam2Processor,
)

from hair_sam2_dataset import (
    HairSAM2Dataset,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# GPU SETTINGS
# ============================================================

def configure_gpu():

    if torch.cuda.is_available():

        # Useful on Ampere GPUs such as A40.
        # Harmless on T4.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.set_float32_matmul_precision(
            "high"
        )


# ============================================================
# MIXED PRECISION
# ============================================================

def configure_mixed_precision(device):

    if device.type != "cuda":

        print("AMP disabled: CPU")

        return (
            torch.float32,
            False,
        )

    major, minor = (
        torch.cuda.get_device_capability()
    )

    print(
        f"GPU compute capability: "
        f"{major}.{minor}"
    )

    # --------------------------------------------------------
    # Ampere or newer:
    # native BF16 support
    #
    # T4:
    # compute capability 7.5 -> FP16
    # --------------------------------------------------------

    if major >= 8:

        print(
            "AMP: bfloat16 "
            "(native hardware support)"
        )

        return (
            torch.bfloat16,
            False,
        )

    else:

        print(
            "AMP: float16"
        )

        return (
            torch.float16,
            True,
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def build_collate_fn(processor):

    def collate_fn(batch):

        images = [
            item["image"]
            for item in batch
        ]

        # One object / one box per image.
        input_boxes = [
            [
                item[
                    "box"
                ].tolist()
            ]
            for item in batch
        ]

        processed = processor(
            images=images,
            input_boxes=input_boxes,
            return_tensors="pt",
        )

        gt_masks = torch.stack(
            [
                item["mask"]
                for item in batch
            ],
            dim=0,
        )

        return {
            "pixel_values":
                processed[
                    "pixel_values"
                ],

            "input_boxes":
                processed[
                    "input_boxes"
                ],

            "gt_masks":
                gt_masks,
        }

    return collate_fn


# ============================================================
# NORMALIZE SAM2 OUTPUT SHAPE
# ============================================================

def normalize_pred_masks(pred_masks):

    """
    Convert SAM2 output into:

        B x 1 x H x W
    """

    if pred_masks.ndim == 5:

        pred_masks = (
            pred_masks[
                :,
                0,
                0,
                :,
                :
            ]
            .unsqueeze(1)
        )

    elif pred_masks.ndim == 4:

        pass

    elif pred_masks.ndim == 3:

        pred_masks = (
            pred_masks.unsqueeze(1)
        )

    else:

        raise RuntimeError(
            "Unexpected SAM2 mask shape: "
            f"{pred_masks.shape}"
        )

    return pred_masks


# ============================================================
# FOCAL LOSS
# ============================================================

def binary_focal_loss(
    logits,
    targets,
    alpha=0.5,
    gamma=2.0,
):

    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
    )

    probability = (
        torch.sigmoid(logits)
    )

    p_t = (
        probability * targets
        +
        (1.0 - probability)
        * (1.0 - targets)
    )

    alpha_t = (
        alpha * targets
        +
        (1.0 - alpha)
        * (1.0 - targets)
    )

    focal_weight = (
        alpha_t
        * (
            1.0 - p_t
        ).pow(gamma)
    )

    return (
        focal_weight
        * bce
    ).mean()


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    eps=1e-6,
):

    probabilities = (
        torch.sigmoid(logits)
    )

    probabilities = (
        probabilities.flatten(1)
    )

    targets = (
        targets.flatten(1)
    )

    intersection = (
        probabilities
        * targets
    ).sum(
        dim=1
    )

    denominator = (
        probabilities.sum(
            dim=1
        )
        +
        targets.sum(
            dim=1
        )
    )

    dice = (
        2.0 * intersection
        + eps
    ) / (
        denominator
        + eps
    )

    return (
        1.0 - dice
    ).mean()


# ============================================================
# TOTAL LOSS
# ============================================================

def segmentation_loss(
    logits,
    targets,
):

    focal = binary_focal_loss(
        logits,
        targets,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    total = (
        20.0 * focal
        + dice
    )

    return (
        total,
        focal,
        dice,
    )


# ============================================================
# METRICS
# ============================================================

def update_metrics(
    logits,
    targets,
    metric_state,
    threshold=0.5,
):

    probabilities = (
        torch.sigmoid(logits)
    )

    prediction = (
        probabilities
        >= threshold
    )

    target = (
        targets >= 0.5
    )

    tp = (
        prediction
        & target
    ).sum().item()

    fp = (
        prediction
        & ~target
    ).sum().item()

    fn = (
        ~prediction
        & target
    ).sum().item()

    metric_state["tp"] += tp
    metric_state["fp"] += fp
    metric_state["fn"] += fn


def calculate_metrics(state):

    tp = state["tp"]
    fp = state["fp"]
    fn = state["fn"]

    eps = 1e-8

    precision = (
        tp
        /
        (
            tp + fp + eps
        )
    )

    recall = (
        tp
        /
        (
            tp + fn + eps
        )
    )

    dice = (
        2 * tp
        /
        (
            2 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        /
        (
            tp
            + fp
            + fn
            + eps
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


# ============================================================
# SINGLE BINARY MASK METRICS
# ============================================================

def binary_mask_metrics(
    prediction,
    target,
):

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(target),
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        target,
    ).sum()

    eps = 1e-8

    precision = (
        tp
        / (
            tp + fp + eps
        )
    )

    recall = (
        tp
        / (
            tp + fn + eps
        )
    )

    dice = (
        2 * tp
        / (
            2 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    return {
        "precision":
            float(precision),

        "recall":
            float(recall),

        "dice":
            float(dice),

        "iou":
            float(iou),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    amp_dtype,
):

    model.eval()

    total_loss = 0.0

    metric_state = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

        pixel_values = (
            batch[
                "pixel_values"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        input_boxes = (
            batch[
                "input_boxes"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        gt_masks = (
            batch[
                "gt_masks"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(
                device.type
                == "cuda"
            ),
        ):

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            logits = (
                normalize_pred_masks(
                    outputs.pred_masks
                )
            )

            # Preserve full GT resolution.
            logits = F.interpolate(
                logits,
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            (
                loss,
                _,
                _,
            ) = segmentation_loss(
                logits,
                gt_masks,
            )

        total_loss += (
            loss.item()
        )

        update_metrics(
            logits,
            gt_masks,
            metric_state,
        )

    metrics = calculate_metrics(
        metric_state
    )

    metrics["loss"] = (
        total_loss
        /
        max(
            1,
            len(loader),
        )
    )

    return metrics


# ============================================================
# MASK OVERLAY
# ============================================================

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.45,
):

    output = (
        image.astype(
            np.float32
        ).copy()
    )

    mask = mask.astype(
        bool
    )

    color_array = np.array(
        color,
        dtype=np.float32,
    )

    output[mask] = (
        (
            1.0 - alpha
        )
        * output[mask]
        +
        alpha
        * color_array
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# PANEL LABEL
# ============================================================

def add_panel_label(
    image,
    text,
):

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        text,
        (15, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# CREATE INFERENCE VISUALIZATION
# ============================================================

def create_inference_visualization(
    image,
    gt_mask,
    pred_mask,
    box,
    metrics,
):

    # --------------------------------------------------------
    # PANEL 1: Input + box prompt
    # --------------------------------------------------------

    input_panel = (
        image.copy()
    )

    x1, y1, x2, y2 = (
        box.astype(int)
    )

    cv2.rectangle(
        input_panel,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        3,
    )

    input_panel = (
        add_panel_label(
            input_panel,
            "Input + Box Prompt",
        )
    )

    # --------------------------------------------------------
    # PANEL 2: Ground truth
    # --------------------------------------------------------

    gt_panel = overlay_mask(
        image,
        gt_mask,
        color=(
            0,
            255,
            0,
        ),
    )

    gt_panel = add_panel_label(
        gt_panel,
        "Ground Truth",
    )

    # --------------------------------------------------------
    # PANEL 3: Prediction
    # --------------------------------------------------------

    pred_panel = overlay_mask(
        image,
        pred_mask,
        color=(
            255,
            0,
            0,
        ),
    )

    pred_panel = add_panel_label(
        pred_panel,
        "Prediction",
    )

    # --------------------------------------------------------
    # PANEL 4:
    #
    # Green = TP
    # Red   = FP
    # Blue  = FN
    # --------------------------------------------------------

    comparison = image.copy()

    target = gt_mask.astype(
        bool
    )

    prediction = pred_mask.astype(
        bool
    )

    tp = np.logical_and(
        target,
        prediction,
    )

    fp = np.logical_and(
        prediction,
        np.logical_not(target),
    )

    fn = np.logical_and(
        target,
        np.logical_not(prediction),
    )

    comparison = overlay_mask(
        comparison,
        tp,
        (
            0,
            255,
            0,
        ),
        alpha=0.65,
    )

    comparison = overlay_mask(
        comparison,
        fp,
        (
            255,
            0,
            0,
        ),
        alpha=0.70,
    )

    comparison = overlay_mask(
        comparison,
        fn,
        (
            0,
            0,
            255,
        ),
        alpha=0.70,
    )

    label = (
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"D={metrics['dice']:.3f}  "
        f"IoU={metrics['iou']:.3f}"
    )

    comparison = add_panel_label(
        comparison,
        label,
    )

    # --------------------------------------------------------
    # Join horizontally
    # --------------------------------------------------------

    result = np.concatenate(
        [
            input_panel,
            gt_panel,
            pred_panel,
            comparison,
        ],
        axis=1,
    )

    return result


# ============================================================
# TEST INFERENCE
# ============================================================

@torch.inference_mode()
def run_test_inference(
    model,
    processor,
    images_csv,
    instances_csv,
    output_dir,
    device,
    amp_dtype,
    num_images=20,
    mask_threshold=0.0,
    seed=42,
):

    print()
    print("=" * 80)
    print("RUNNING TEST INFERENCE")
    print("=" * 80)

    output_dir = Path(
        output_dir
    )

    visualization_dir = (
        output_dir
        / "visualizations"
    )

    mask_dir = (
        output_dir
        / "predicted_masks"
    )

    gt_dir = (
        output_dir
        / "ground_truth_masks"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Deterministic TEST dataset.
    #
    # training=False:
    # no scale augmentation
    # no flip
    # no brightness augmentation
    # no box jitter
    # --------------------------------------------------------

    test_dataset = (
        HairSAM2Dataset(
            images_csv=images_csv,
            instances_csv=instances_csv,
            split="test",
            crop_size=1024,
            scale_min=1.0,
            scale_max=1.0,
            box_jitter=0,
            max_instances=None,
            seed=seed,
            training=False,
        )
    )

    test_df = (
        test_dataset.df
    )

    unique_images = (
        test_df[
            "image_name"
        ]
        .drop_duplicates()
        .tolist()
    )

    rng = np.random.default_rng(
        seed
    )

    num_images = min(
        num_images,
        len(unique_images),
    )

    selected_images = (
        rng.choice(
            unique_images,
            size=num_images,
            replace=False,
        )
    )

    # --------------------------------------------------------
    # Select ONE random hair instance from each selected
    # test image.
    # --------------------------------------------------------

    selected_indices = []

    for image_name in selected_images:

        candidate_indices = (
            test_df.index[
                test_df[
                    "image_name"
                ]
                == image_name
            ]
            .to_numpy()
        )

        selected_index = int(
            rng.choice(
                candidate_indices
            )
        )

        selected_indices.append(
            selected_index
        )

    model.eval()

    metric_records = []

    for output_index, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):

        sample = (
            test_dataset[
                dataset_index
            ]
        )

        row = (
            test_df.iloc[
                dataset_index
            ]
        )

        image = sample[
            "image"
        ]

        box = (
            sample[
                "box"
            ]
            .numpy()
            .astype(
                np.float32
            )
        )

        gt_mask = (
            sample[
                "mask"
            ]
            .squeeze(0)
            .numpy()
            .astype(
                np.uint8
            )
        )

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        processed = processor(
            images=[image],
            input_boxes=[
                [
                    box.tolist()
                ]
            ],
            return_tensors="pt",
        )

        pixel_values = (
            processed[
                "pixel_values"
            ]
            .to(device)
        )

        input_boxes = (
            processed[
                "input_boxes"
            ]
            .to(device)
        )

        # Keep original sizes for post-processing.
        original_sizes = (
            processed[
                "original_sizes"
            ]
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=(
                device.type
                == "cuda"
            ),
        ):

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

        # ----------------------------------------------------
        # Convert decoder mask back to input resolution.
        #
        # HF post_process_masks handles SAM2's mask decoder
        # resolution and returns masks at original input size.
        # ----------------------------------------------------

        processed_masks = (
            processor.post_process_masks(
                outputs.pred_masks
                .detach()
                .cpu(),
                original_sizes,
                mask_threshold=(
                    mask_threshold
                ),
                binarize=True,
            )
        )

        pred_mask = (
            processed_masks[0]
            .squeeze()
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if (
            pred_mask.shape
            != gt_mask.shape
        ):

            pred_mask = cv2.resize(
                pred_mask,
                (
                    gt_mask.shape[1],
                    gt_mask.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = binary_mask_metrics(
            pred_mask,
            gt_mask,
        )

        # Predicted SAM IoU confidence
        predicted_iou = float(
            outputs.iou_scores
            .detach()
            .float()
            .cpu()
            .reshape(-1)[0]
            .item()
        )

        metrics_record = {
            "image_name":
                row[
                    "image_name"
                ],

            "instance_id":
                int(
                    row[
                        "instance_id"
                    ]
                ),

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "dice":
                metrics[
                    "dice"
                ],

            "iou":
                metrics[
                    "iou"
                ],

            "sam_predicted_iou":
                predicted_iou,
        }

        metric_records.append(
            metrics_record
        )

        # ----------------------------------------------------
        # Create visualization
        # ----------------------------------------------------

        visualization = (
            create_inference_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                box=box,
                metrics=metrics,
            )
        )

        stem = (
            f"{output_index:03d}_"
            f"{Path(row['image_name']).stem}_"
            f"instance_{int(row['instance_id'])}"
        )

        # ----------------------------------------------------
        # Save comparison
        #
        # image is RGB internally.
        # cv2.imwrite expects BGR.
        # ----------------------------------------------------

        visualization_bgr = (
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR,
            )
        )

        cv2.imwrite(
            str(
                visualization_dir
                / f"{stem}_comparison.jpg"
            ),
            visualization_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        # ----------------------------------------------------
        # Save binary prediction separately
        # ----------------------------------------------------

        cv2.imwrite(
            str(
                mask_dir
                / f"{stem}_pred.png"
            ),
            pred_mask * 255,
        )

        # ----------------------------------------------------
        # Save GT mask separately
        # ----------------------------------------------------

        cv2.imwrite(
            str(
                gt_dir
                / f"{stem}_gt.png"
            ),
            gt_mask * 255,
        )

        print(
            f"[{output_index}/{num_images}] "
            f"{row['image_name']} | "
            f"instance={int(row['instance_id'])} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f}"
        )

    # ========================================================
    # SAVE TEST INFERENCE METRICS
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_records
    )

    metrics_csv = (
        output_dir
        / "inference_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary = {
        "num_test_images":
            int(
                len(
                    metrics_df
                )
            ),

        "mean_precision":
            float(
                metrics_df[
                    "precision"
                ].mean()
            ),

        "mean_recall":
            float(
                metrics_df[
                    "recall"
                ].mean()
            ),

        "mean_dice":
            float(
                metrics_df[
                    "dice"
                ].mean()
            ),

        "mean_iou":
            float(
                metrics_df[
                    "iou"
                ].mean()
            ),

        "median_precision":
            float(
                metrics_df[
                    "precision"
                ].median()
            ),

        "median_recall":
            float(
                metrics_df[
                    "recall"
                ].median()
            ),

        "median_dice":
            float(
                metrics_df[
                    "dice"
                ].median()
            ),

        "median_iou":
            float(
                metrics_df[
                    "iou"
                ].median()
            ),

        "mask_logit_threshold":
            float(
                mask_threshold
            ),
    }

    summary_path = (
        output_dir
        / "inference_summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 80)
    print("TEST INFERENCE SUMMARY")
    print("=" * 80)

    print(
        f"Images     : "
        f"{summary['num_test_images']}"
    )

    print(
        f"Precision  : "
        f"{summary['mean_precision']:.6f}"
    )

    print(
        f"Recall     : "
        f"{summary['mean_recall']:.6f}"
    )

    print(
        f"Dice       : "
        f"{summary['mean_dice']:.6f}"
    )

    print(
        f"IoU        : "
        f"{summary['mean_iou']:.6f}"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main(args):

    seed_everything(
        args.seed
    )

    configure_gpu()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ========================================================
    # MODEL + PROCESSOR
    # ========================================================

    print(
        "\nLoading SAM2.1..."
    )

    processor = (
        Sam2Processor
        .from_pretrained(
            args.model
        )
    )

    model = (
        Sam2Model
        .from_pretrained(
            args.model
        )
    )

    # ========================================================
    # FREEZE / UNFREEZE
    # ========================================================

    if args.train_encoder:

        print(
            "Training full model."
        )

        for parameter in (
            model.parameters()
        ):

            parameter.requires_grad = (
                True
            )

    else:

        print(
            "Freezing vision encoder."
        )

        for parameter in (
            model.vision_encoder
            .parameters()
        ):

            parameter.requires_grad = (
                False
            )

        for parameter in (
            model.prompt_encoder
            .parameters()
        ):

            parameter.requires_grad = (
                True
            )

        for parameter in (
            model.mask_decoder
            .parameters()
        ):

            parameter.requires_grad = (
                True
            )

    model = model.to(
        device
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in trainable_params
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    (
        amp_dtype,
        use_scaler,
    ) = configure_mixed_precision(
        device
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        HairSAM2Dataset(
            images_csv=args.images_csv,
            instances_csv=args.instances_csv,
            split="train",
            crop_size=1024,
            scale_min=0.85,
            scale_max=1.15,
            box_jitter=args.box_jitter,
            max_instances=(
                args.max_train_instances
            ),
            seed=args.seed,
            training=True,
        )
    )

    val_dataset = (
        HairSAM2Dataset(
            images_csv=args.images_csv,
            instances_csv=args.instances_csv,
            split="val",
            crop_size=1024,
            scale_min=1.0,
            scale_max=1.0,
            box_jitter=0,
            max_instances=(
                args.max_val_instances
            ),
            seed=args.seed,
            training=False,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    collate_fn = (
        build_collate_fn(
            processor
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            args.workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=(
            args.workers > 0
        ),
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / args.accumulation_steps
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * args.epochs
    )

    warmup_steps = min(
        args.warmup_steps,
        max(
            1,
            total_optimizer_steps
            // 10,
        ),
    )

    # ========================================================
    # WARMUP + COSINE SCHEDULER
    # ========================================================

    def lr_lambda(step):

        if step < warmup_steps:

            return (
                step + 1
            ) / max(
                1,
                warmup_steps,
            )

        progress = (
            step
            - warmup_steps
        ) / max(
            1,
            total_optimizer_steps
            - warmup_steps,
        )

        progress = min(
            progress,
            1.0,
        )

        return (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    scheduler = (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_lambda,
        )
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    best_dice = -1.0

    best_dir = os.path.join(
        args.output_dir,
        "best_model",
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        # If encoder is frozen, also keep it in eval mode.
        if not args.train_encoder:

            model.vision_encoder.eval()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{args.epochs}"
            ),
        )

        for (
            batch_idx,
            batch,
        ) in enumerate(
            progress,
            start=1,
        ):

            pixel_values = (
                batch[
                    "pixel_values"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            input_boxes = (
                batch[
                    "input_boxes"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            gt_masks = (
                batch[
                    "gt_masks"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False,
                )

                logits = (
                    normalize_pred_masks(
                        outputs.pred_masks
                    )
                )

                logits = F.interpolate(
                    logits,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                (
                    loss,
                    focal,
                    dice,
                ) = segmentation_loss(
                    logits,
                    gt_masks,
                )

                loss_for_backward = (
                    loss
                    / args.accumulation_steps
                )

            # =================================================
            # BACKWARD
            # =================================================

            if use_scaler:

                scaler.scale(
                    loss_for_backward
                ).backward()

            else:

                loss_for_backward.backward()

            # =================================================
            # OPTIMIZER STEP
            # =================================================

            if (
                batch_idx
                % args.accumulation_steps
                == 0

                or

                batch_idx
                == len(train_loader)
            ):

                if use_scaler:

                    scaler.unscale_(
                        optimizer
                    )

                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=1.0,
                )

                if use_scaler:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

            running_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=(
                    f"{loss.item():.4f}"
                ),
                focal=(
                    f"{focal.item():.4f}"
                ),
                dice_loss=(
                    f"{dice.item():.4f}"
                ),
                lr=(
                    f"{optimizer.param_groups[0]['lr']:.2e}"
                ),
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
        )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"EPOCH {epoch}"
        )

        print(
            "=" * 70
        )

        print(
            f"Train Loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val Loss   : "
            f"{metrics['loss']:.6f}"
        )

        print(
            f"Precision  : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall     : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"Dice       : "
            f"{metrics['dice']:.6f}"
        )

        print(
            f"IoU        : "
            f"{metrics['iou']:.6f}"
        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if (
            metrics["dice"]
            > best_dice
        ):

            best_dice = (
                metrics["dice"]
            )

            model.save_pretrained(
                best_dir
            )

            processor.save_pretrained(
                best_dir
            )

            print(
                "\nSaved new best model:"
            )

            print(
                best_dir
            )

        print()

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_dir = os.path.join(
        args.output_dir,
        "last_model",
    )

    model.save_pretrained(
        final_dir
    )

    processor.save_pretrained(
        final_dir
    )

    print()
    print(
        "Training complete."
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.6f}"
    )

    # ========================================================
    # REMOVE CURRENT MODEL FROM GPU BEFORE RELOADING BEST
    # ========================================================

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # LOAD BEST MODEL FOR TEST INFERENCE
    # ========================================================

    if args.num_test_inferences > 0:

        print()
        print(
            "Loading BEST checkpoint "
            "for test inference..."
        )

        inference_processor = (
            Sam2Processor
            .from_pretrained(
                best_dir
            )
        )

        inference_model = (
            Sam2Model
            .from_pretrained(
                best_dir
            )
            .to(device)
        )

        inference_output_dir = (
            Path(
                args.output_dir
            )
            / "test_inference"
        )

        run_test_inference(
            model=inference_model,
            processor=inference_processor,
            images_csv=args.images_csv,
            instances_csv=args.instances_csv,
            output_dir=(
                inference_output_dir
            ),
            device=device,
            amp_dtype=amp_dtype,
            num_images=(
                args.num_test_inferences
            ),
            mask_threshold=(
                args.inference_mask_threshold
            ),
            seed=args.seed,
        )


# ============================================================
# ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--instances_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "facebook/"
            "sam2.1-hiera-base-plus"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "sam2_hair_runs/"
            "baseline_base_plus"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--box_jitter",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_train_instances",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_val_instances",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--train_encoder",
        action="store_true",
    )

    # --------------------------------------------------------
    # Test inference options
    # --------------------------------------------------------

    parser.add_argument(
        "--num_test_inferences",
        type=int,
        default=20,
        help=(
            "Number of random TEST images to visualize "
            "after training. One random shaft is selected "
            "from each image. Set 0 to disable."
        ),
    )

    parser.add_argument(
        "--inference_mask_threshold",
        type=float,
        default=0.0,
        help=(
            "SAM2 mask-logit threshold. "
            "0.0 corresponds approximately to "
            "probability 0.5."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)