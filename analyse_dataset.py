import cv2
import json
import random
import hashlib
import argparse
import re
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# FIND IMAGE-LABEL PAIRS
# ============================================================

def find_image_label_pairs(images_dir, labels_dir):
    """
    Find images with corresponding YOLO segmentation labels.

    Expected:
        images/image001.jpg
        labels/image001.txt
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory does not exist: {images_dir}"
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory does not exist: {labels_dir}"
        )

    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    pairs = []
    missing_labels = []

    for image_path in image_paths:

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():

            pairs.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

        else:

            missing_labels.append(
                image_path
            )

    return pairs, missing_labels


# ============================================================
# PARSE YOLO SEGMENTATION LABEL
# ============================================================

def parse_yolo_segmentation(
    label_path,
    image_width,
    image_height,
):
    """
    Parse YOLO segmentation labels.

    Format:
        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized relative to the full
    image width and height.

    Returns:
        instances
        parser_stats
        issues
    """

    instances = []

    parser_stats = {
        "total_lines": 0,
        "empty_lines": 0,
        "malformed_lines": 0,
        "out_of_range_polygons": 0,
        "valid_polygons": 0,
    }

    issues = []

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        lines = f.readlines()

    for line_idx, line in enumerate(
        lines,
        start=1,
    ):

        parser_stats["total_lines"] += 1

        line = line.strip()

        # ----------------------------------------------------
        # Empty line
        # ----------------------------------------------------

        if not line:

            parser_stats[
                "empty_lines"
            ] += 1

            continue

        values = line.split()

        # Need:
        # class_id + at least 3 polygon points
        # = 1 + 6 values
        if len(values) < 7:

            parser_stats[
                "malformed_lines"
            ] += 1

            issues.append(
                {
                    "line": line_idx,
                    "reason": (
                        "Too few values for polygon"
                    ),
                }
            )

            continue

        try:

            class_id = int(
                float(values[0])
            )

            coords = np.array(
                list(
                    map(
                        float,
                        values[1:],
                    )
                ),
                dtype=np.float32,
            )

        except ValueError:

            parser_stats[
                "malformed_lines"
            ] += 1

            issues.append(
                {
                    "line": line_idx,
                    "reason": (
                        "Non-numeric annotation value"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Must have x,y pairs
        # ----------------------------------------------------

        if len(coords) % 2 != 0:

            parser_stats[
                "malformed_lines"
            ] += 1

            issues.append(
                {
                    "line": line_idx,
                    "reason": (
                        "Odd number of polygon coordinates"
                    ),
                }
            )

            continue

        coords = coords.reshape(
            -1,
            2,
        )

        if len(coords) < 3:

            parser_stats[
                "malformed_lines"
            ] += 1

            issues.append(
                {
                    "line": line_idx,
                    "reason": (
                        "Fewer than 3 polygon points"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Check normalized range
        # ----------------------------------------------------

        out_of_range = (
            np.any(coords < 0.0)
            or np.any(coords > 1.0)
        )

        if out_of_range:

            parser_stats[
                "out_of_range_polygons"
            ] += 1

            issues.append(
                {
                    "line": line_idx,
                    "reason": (
                        "Coordinates outside [0,1]; "
                        "coordinates were clipped"
                    ),
                }
            )

            coords = np.clip(
                coords,
                0.0,
                1.0,
            )

        # ----------------------------------------------------
        # Convert YOLO normalized coordinates
        # to pixel coordinates.
        #
        # IMPORTANT:
        # Multiply by full width/height, NOT width-1.
        #
        # Values exactly equal to 1.0 are clipped afterward
        # into the valid pixel-index range.
        # ----------------------------------------------------

        polygon = coords.copy()

        polygon[:, 0] *= image_width
        polygon[:, 1] *= image_height

        polygon = np.round(
            polygon
        ).astype(np.int32)

        polygon[:, 0] = np.clip(
            polygon[:, 0],
            0,
            image_width - 1,
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1],
            0,
            image_height - 1,
        )

        parser_stats[
            "valid_polygons"
        ] += 1

        instances.append(
            {
                "class_id": class_id,
                "polygon": polygon,
                "line_idx": line_idx,
            }
        )

    return (
        instances,
        parser_stats,
        issues,
    )


# ============================================================
# POLYGON -> BINARY MASK
# ============================================================

def polygon_to_mask(
    polygon,
    image_height,
    image_width,
):
    """
    Convert one polygon into a binary instance mask.

    Background = 0
    Hair       = 1
    """

    mask = np.zeros(
        (
            image_height,
            image_width,
        ),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [polygon],
        1,
    )

    return mask


# ============================================================
# GET BOUNDING BOX
# ============================================================

def get_bbox_from_mask(mask):
    """
    Get bounding box:

        [x_min, y_min, x_max, y_max]
    """

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


# ============================================================
# POSITIVE PROMPT POINT
# ============================================================

def get_positive_point_and_width(mask, bbox, pad=2):
    """
    Single distance-transform pass, cropped to the instance's
    bounding box (padded), reused for both the SAM prompt point
    and the max local width estimate.

    Returns:
        positive_point : [x, y] in full-image coordinates, or None
        max_local_width : float
    """

    x1, y1, x2, y2 = bbox
    h, w = mask.shape

    x1c = max(x1 - pad, 0)
    y1c = max(y1 - pad, 0)
    x2c = min(x2 + pad, w - 1)
    y2c = min(y2 + pad, h - 1)

    crop_255 = (
        mask[y1c:y2c + 1, x1c:x2c + 1] * 255
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        crop_255,
        cv2.DIST_L2,
        5,
    )

    _, max_value, _, max_location = cv2.minMaxLoc(distance)

    if max_value <= 0:
        positive_point = None
    else:
        positive_point = [
            int(max_location[0] + x1c),
            int(max_location[1] + y1c),
        ]

    max_local_width = 2.0 * float(distance.max())

    return positive_point, max_local_width


# ============================================================
# INSTANCE STATISTICS
# ============================================================

def calculate_instance_stats(
    mask,
    bbox,
):
    """
    Calculate statistics for an individual hair shaft.

    max_local_width_px is an approximation based on
    twice the maximum distance-transform radius.

    It is NOT the average shaft thickness.
    """

    area = int(
        mask.sum()
    )

    x1, y1, x2, y2 = bbox

    bbox_width = (
        x2 - x1 + 1
    )

    bbox_height = (
        y2 - y1 + 1
    )

    return {
        "area_pixels": area,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
    }


# ============================================================
# BORDER CHECK
# ============================================================

def touches_border(mask):
    """
    Check whether instance touches image boundary.
    """

    return bool(
        np.any(mask[0, :] > 0)
        or np.any(mask[-1, :] > 0)
        or np.any(mask[:, 0] > 0)
        or np.any(mask[:, -1] > 0)
    )


# ============================================================
# OPTIONAL PATIENT/SESSION GROUP EXTRACTION
# ============================================================

def extract_group_id(
    image_path,
    group_regex=None,
):
    """
    Extract patient/session/group ID from filename.

    If group_regex is None:
        every image is its own group.

    Example filename:
        patient12_session03_image001.jpg

    Example regex:
        r'^(patient\\d+)'

    -> group_id = patient12

    If regex contains a capture group, the first
    capture group is used.
    """

    if group_regex is None:

        return image_path.name

    match = re.search(
        group_regex,
        image_path.stem,
    )

    if match is None:

        raise ValueError(
            f"group_regex did not match image: "
            f"{image_path.name}"
        )

    if match.groups():

        return match.group(1)

    return match.group(0)


# ============================================================
# DATASET SPLIT
# ============================================================

def split_dataset(
    pairs,
    train_ratio,
    val_ratio,
    test_ratio,
    seed,
    group_regex=None,
):
    """
    Split dataset while optionally keeping images from
    the same patient/session together.

    Without group_regex:
        image-level random split.

    With group_regex:
        group-level split.

    Note:
        Group-level ratios may be approximate because an
        entire group must stay inside one split.
    """

    ratio_sum = (
        train_ratio
        + val_ratio
        + test_ratio
    )

    if not np.isclose(
        ratio_sum,
        1.0,
    ):

        raise ValueError(
            "train_ratio + val_ratio + test_ratio "
            "must equal 1.0"
        )

    rng = random.Random(
        seed
    )

    # --------------------------------------------------------
    # Create groups
    # --------------------------------------------------------

    groups = {}

    for original_pair in pairs:

        pair = original_pair.copy()

        group_id = extract_group_id(
            pair["image_path"],
            group_regex,
        )

        pair["group_id"] = group_id

        groups.setdefault(
            group_id,
            [],
        ).append(pair)

    group_items = list(
        groups.items()
    )

    rng.shuffle(
        group_items
    )

    group_items.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    total_images = len(pairs)

    train_target = (
        total_images
        * train_ratio
    )

    val_target = (
        total_images
        * val_ratio
    )

    split_data = {
        "train": [],
        "val": [],
        "test": [],
    }

    # --------------------------------------------------------
    # For ordinary image-level splitting, this behaves
    # approximately like the original random shuffle.
    #
    # For groups, keep complete groups together.
    # --------------------------------------------------------

    for _, group_pairs in group_items:

        train_deficit = (
            train_target
            - len(
                split_data["train"]
            )
        )

        val_deficit = (
            val_target
            - len(
                split_data["val"]
            )
        )

        # Train still needs most images
        if (
            train_deficit > 0
            and train_deficit
            >= val_deficit
        ):

            split_data[
                "train"
            ].extend(
                group_pairs
            )

        elif val_deficit > 0:

            split_data[
                "val"
            ].extend(
                group_pairs
            )

        else:

            split_data[
                "test"
            ].extend(
                group_pairs
            )

    return (
        split_data["train"],
        split_data["val"],
        split_data["test"],
    )


# ============================================================
# SAVE SPLIT FILE
# ============================================================

def save_split_file(
    split_pairs,
    output_path,
):
    """
    Save:
        image_path<TAB>label_path<TAB>group_id
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for pair in split_pairs:

            f.write(
                f"{pair['image_path']}\t"
                f"{pair['label_path']}\t"
                f"{pair['group_id']}\n"
            )


# ============================================================
# DETERMINISTIC VISUALIZATION SEED
# ============================================================

def deterministic_seed(
    text,
):
    """
    Python's built-in hash() is intentionally randomized
    between processes.

    Use MD5 here only for deterministic visualization
    colors, not for cryptographic purposes.
    """

    digest = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    return (
        int(
            digest,
            16,
        )
        % (2 ** 32)
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_image(
    image_path,
    label_path,
    output_path,
):
    """
    Draw:
        - filled ground-truth instance masks
        - polygon boundaries
        - SAM bounding-box prompts
        - SAM positive-point prompts
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "[WARNING] Could not read "
            f"preview image: {image_path}"
        )

        return False

    h, w = image.shape[:2]

    (
        instances,
        _,
        _,
    ) = parse_yolo_segmentation(
        label_path,
        w,
        h,
    )

    overlay = image.copy()

    seed = deterministic_seed(
        image_path.name
    )

    rng = np.random.default_rng(
        seed
    )

    colors = []

    # --------------------------------------------------------
    # Filled masks
    # --------------------------------------------------------

    for instance in instances:

        polygon = instance[
            "polygon"
        ]

        color = tuple(
            int(x)
            for x in rng.integers(
                50,
                255,
                size=3,
            )
        )

        colors.append(
            color
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            color,
        )

    visualized = cv2.addWeighted(
        overlay,
        0.40,
        image,
        0.60,
        0,
    )

    valid_instance_count = 0

    # --------------------------------------------------------
    # Boundaries, boxes, prompt points
    # --------------------------------------------------------

    for instance_idx, instance in enumerate(
        instances
    ):

        polygon = instance[
            "polygon"
        ]

        color = colors[
            instance_idx
        ]

        mask = polygon_to_mask(
            polygon,
            h,
            w,
        )

        bbox = get_bbox_from_mask(
            mask
        )

        if bbox is None:
            continue

        valid_instance_count += 1

        positive_point, _ = (
            get_positive_point_and_width(
                mask,
                bbox,
            )
        )

        # Polygon boundary
        cv2.polylines(
            visualized,
            [polygon],
            isClosed=True,
            color=color,
            thickness=2,
        )

        # Bounding box
        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            visualized,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        # Positive point
        if positive_point is not None:

            px, py = positive_point

            # White center
            cv2.circle(
                visualized,
                (px, py),
                4,
                (255, 255, 255),
                -1,
            )

            # Black outline
            cv2.circle(
                visualized,
                (px, py),
                6,
                (0, 0, 0),
                1,
            )

    text = (
        f"Hair instances: "
        f"{valid_instance_count}"
    )

    cv2.putText(
        visualized,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(
        str(output_path),
        visualized,
    )

    if not success:

        print(
            "[WARNING] Could not save preview: "
            f"{output_path}"
        )

        return False

    return True


# ============================================================
# ADD DISTRIBUTION PERCENTILES
# ============================================================

def add_percentiles(
    summary,
    instance_df,
):
    """
    Add useful tail statistics.

    Thin-tail statistics are particularly important for
    hair shafts because small widths may disappear after
    image resizing.
    """

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    for p in percentiles:

        summary[
            f"max_local_width_p{p}"
        ] = float(
            np.percentile(
                instance_df[
                    "max_local_width_px"
                ],
                p,
            )
        )

        summary[
            f"area_pixels_p{p}"
        ] = float(
            np.percentile(
                instance_df[
                    "area_pixels"
                ],
                p,
            )
        )

        summary[
            f"bbox_width_p{p}"
        ] = float(
            np.percentile(
                instance_df[
                    "bbox_width"
                ],
                p,
            )
        )

        summary[
            f"bbox_height_p{p}"
        ] = float(
            np.percentile(
                instance_df[
                    "bbox_height"
                ],
                p,
            )
        )


# ============================================================
# ADD THIN-INSTANCE COUNTS
# ============================================================

def add_thin_instance_statistics(
    summary,
    instance_df,
):
    """
    Count approximate shaft widths below important
    thresholds.
    """

    thresholds = [
        1,
        2,
        3,
        4,
        5,
        8,
        10,
    ]

    total_instances = len(
        instance_df
    )

    for threshold in thresholds:

        count = int(
            (
                instance_df[
                    "max_local_width_px"
                ]
                <= threshold
            ).sum()
        )

        fraction = (
            count
            / total_instances
        )

        summary[
            f"instances_width_le_{threshold}px"
        ] = count

        summary[
            f"fraction_width_le_{threshold}px"
        ] = float(
            fraction
        )


# ============================================================
# MAIN
# ============================================================

def main(args):

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    images_dir = Path(
        args.images_dir
    )

    labels_dir = Path(
        args.labels_dir
    )

    output_dir = Path(
        args.output_dir
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    splits_dir = (
        output_dir
        / "splits"
    )

    audit_dir = (
        output_dir
        / "audit"
    )

    preview_dir = (
        output_dir
        / "previews"
    )

    splits_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # FIND IMAGE-LABEL PAIRS
    # ========================================================

    (
        pairs,
        missing_labels,
    ) = find_image_label_pairs(
        images_dir,
        labels_dir,
    )

    print()
    print("=" * 80)
    print("DATASET PAIRING")
    print("=" * 80)

    print(
        f"Valid image-label pairs : "
        f"{len(pairs)}"
    )

    print(
        f"Images without labels   : "
        f"{len(missing_labels)}"
    )

    if len(pairs) == 0:

        raise RuntimeError(
            "No valid image-label pairs found."
        )

    # --------------------------------------------------------
    # Save missing labels
    # --------------------------------------------------------

    with open(
        audit_dir
        / "missing_labels.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for path in missing_labels:

            f.write(
                f"{path}\n"
            )

    # ========================================================
    # SPLIT DATASET
    # ========================================================

    if args.group_regex:

        print()
        print(
            "Grouped splitting ENABLED"
        )

        print(
            f"group_regex: "
            f"{args.group_regex}"
        )

    else:

        print()
        print(
            "Image-level splitting ENABLED"
        )

    (
        train_pairs,
        val_pairs,
        test_pairs,
    ) = split_dataset(
        pairs=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_regex=args.group_regex,
    )

    print()
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    print(
        f"Train images : {len(train_pairs)}"
    )

    print(
        f"Val images   : {len(val_pairs)}"
    )

    print(
        f"Test images  : {len(test_pairs)}"
    )

    print(
        f"Train groups : {len(set(p['group_id'] for p in train_pairs))}"
    )

    print(
        f"Val groups   : {len(set(p['group_id'] for p in val_pairs))}"
    )

    print(
        f"Test groups  : {len(set(p['group_id'] for p in test_pairs))}"
    )

    print(
        f"Train fraction actual: {len(train_pairs)/len(pairs):.3f} (target {args.train_ratio})"
    )

    print(
        f"Val fraction actual:   {len(val_pairs)/len(pairs):.3f} (target {args.val_ratio})"
    )

    print(
        f"Test fraction actual:  {len(test_pairs)/len(pairs):.3f} (target {args.test_ratio})"
    )

    # --------------------------------------------------------
    # Lookup tables
    # --------------------------------------------------------

    split_lookup = {}
    group_lookup = {}

    for pair in train_pairs:

        key = str(
            pair["image_path"]
        )

        split_lookup[key] = "train"

        group_lookup[key] = (
            pair["group_id"]
        )

    for pair in val_pairs:

        key = str(
            pair["image_path"]
        )

        split_lookup[key] = "val"

        group_lookup[key] = (
            pair["group_id"]
        )

    for pair in test_pairs:

        key = str(
            pair["image_path"]
        )

        split_lookup[key] = "test"

        group_lookup[key] = (
            pair["group_id"]
        )

    # --------------------------------------------------------
    # Save split files
    # --------------------------------------------------------

    save_split_file(
        train_pairs,
        splits_dir / "train.txt",
    )

    save_split_file(
        val_pairs,
        splits_dir / "val.txt",
    )

    save_split_file(
        test_pairs,
        splits_dir / "test.txt",
    )

    # ========================================================
    # AUDIT
    # ========================================================

    image_records = []
    instance_records = []

    unreadable_images = []

    annotation_issues = []

    dataset_parser_stats = {
        "total_lines": 0,
        "empty_lines": 0,
        "malformed_lines": 0,
        "out_of_range_polygons": 0,
        "valid_polygons": 0,
    }

    total_degenerate_instances = 0

    print()
    print("=" * 80)
    print("AUDITING DATASET")
    print("=" * 80)

    for idx, pair in enumerate(
        pairs,
        start=1,
    ):

        image_path = pair[
            "image_path"
        ]

        label_path = pair[
            "label_path"
        ]

        image = cv2.imread(
            str(image_path)
        )

        # ----------------------------------------------------
        # Unreadable image
        # ----------------------------------------------------

        if image is None:

            print(
                "[WARNING] Could not read: "
                f"{image_path}"
            )

            unreadable_images.append(
                str(image_path)
            )

            continue

        h, w = image.shape[:2]

        (
            instances,
            parser_stats,
            issues,
        ) = parse_yolo_segmentation(
            label_path,
            w,
            h,
        )

        # ----------------------------------------------------
        # Aggregate parser statistics
        # ----------------------------------------------------

        for key in dataset_parser_stats:

            dataset_parser_stats[
                key
            ] += parser_stats[
                key
            ]

        for issue in issues:

            annotation_issues.append(
                {
                    "image_name":
                        image_path.name,

                    "label_path":
                        str(label_path),

                    "line":
                        issue["line"],

                    "reason":
                        issue["reason"],
                }
            )

        num_parsed_instances = len(
            instances
        )

        num_valid_instances = 0

        num_degenerate_instances = 0

        # ----------------------------------------------------
        # Count map used for:
        #
        # foreground coverage
        # overlapping/crossing annotation analysis
        # ----------------------------------------------------

        count_map = np.zeros(
            (h, w),
            dtype=np.uint16,
        )

        # ====================================================
        # INSTANCE LOOP
        # ====================================================

        for (
            instance_idx,
            instance,
        ) in enumerate(
            instances
        ):

            polygon = instance[
                "polygon"
            ]

            mask = polygon_to_mask(
                polygon,
                h,
                w,
            )

            bbox = get_bbox_from_mask(
                mask
            )

            # ------------------------------------------------
            # Track instances that disappear during
            # rasterization
            # ------------------------------------------------

            if bbox is None:

                num_degenerate_instances += 1

                total_degenerate_instances += 1

                continue

            num_valid_instances += 1

            # Accumulate occupancy
            count_map += (
                mask.astype(
                    np.uint16
                )
            )

            positive_point, max_local_width = (
                get_positive_point_and_width(
                    mask,
                    bbox,
                )
            )

            stats = (
                calculate_instance_stats(
                    mask,
                    bbox,
                )
            )

            border = touches_border(
                mask
            )

            instance_records.append(
                {
                    "image_name":
                        image_path.name,

                    "image_path":
                        str(image_path),

                    "split":
                        split_lookup[
                            str(
                                image_path
                            )
                        ],

                    "group_id":
                        group_lookup[
                            str(
                                image_path
                            )
                        ],

                    "instance_id":
                        instance_idx,

                    "source_label_line":
                        instance[
                            "line_idx"
                        ],

                    "class_id":
                        instance[
                            "class_id"
                        ],

                    "polygon_points":
                        len(polygon),

                    "area_pixels":
                        stats[
                            "area_pixels"
                        ],

                    "bbox_x1":
                        bbox[0],

                    "bbox_y1":
                        bbox[1],

                    "bbox_x2":
                        bbox[2],

                    "bbox_y2":
                        bbox[3],

                    "bbox_width":
                        stats[
                            "bbox_width"
                        ],

                    "bbox_height":
                        stats[
                            "bbox_height"
                        ],

                    "max_local_width_px":
                        max_local_width,

                    "positive_x":
                        (
                            positive_point[0]
                            if positive_point
                            is not None
                            else None
                        ),

                    "positive_y":
                        (
                            positive_point[1]
                            if positive_point
                            is not None
                            else None
                        ),

                    "touches_border":
                        border,
                }
            )

        # ====================================================
        # IMAGE-LEVEL STATISTICS
        # ====================================================

        foreground_pixels = int(
            (
                count_map > 0
            ).sum()
        )

        overlap_pixels = int(
            (
                count_map > 1
            ).sum()
        )

        image_pixels = (
            h * w
        )

        foreground_fraction = (
            foreground_pixels
            / image_pixels
        )

        overlap_fraction_image = (
            overlap_pixels
            / image_pixels
        )

        overlap_fraction_foreground = (
            overlap_pixels
            / foreground_pixels
            if foreground_pixels > 0
            else 0.0
        )

        image_records.append(
            {
                "image_name":
                    image_path.name,

                "image_path":
                    str(image_path),

                "label_path":
                    str(label_path),

                "split":
                    split_lookup[
                        str(
                            image_path
                        )
                    ],

                "group_id":
                    group_lookup[
                        str(
                            image_path
                        )
                    ],

                "width": w,
                "height": h,

                "parsed_instances":
                    num_parsed_instances,

                "valid_instances":
                    num_valid_instances,

                "degenerate_instances":
                    num_degenerate_instances,

                "foreground_pixels":
                    foreground_pixels,

                "foreground_fraction":
                    foreground_fraction,

                "overlap_pixels":
                    overlap_pixels,

                "overlap_fraction_image":
                    overlap_fraction_image,

                "overlap_fraction_foreground":
                    overlap_fraction_foreground,

                "parser_total_lines":
                    parser_stats[
                        "total_lines"
                    ],

                "parser_malformed_lines":
                    parser_stats[
                        "malformed_lines"
                    ],

                "parser_out_of_range_polygons":
                    parser_stats[
                        "out_of_range_polygons"
                    ],
            }
        )

        if (
            idx % 100 == 0
            or idx == len(pairs)
        ):

            print(
                f"Processed "
                f"{idx}/{len(pairs)} images"
            )

    # ========================================================
    # SAVE UNREADABLE IMAGES
    # ========================================================

    with open(
        audit_dir
        / "unreadable_images.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for path in unreadable_images:

            f.write(
                f"{path}\n"
            )

    # ========================================================
    # SAVE ANNOTATION ISSUES
    # ========================================================

    annotation_issues_df = (
        pd.DataFrame(
            annotation_issues
        )
    )

    annotation_issues_df.to_csv(
        audit_dir
        / "annotation_issues.csv",
        index=False,
    )

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    image_df = pd.DataFrame(
        image_records
    )

    instance_df = pd.DataFrame(
        instance_records
    )

    # --------------------------------------------------------
    # Guard against confusing downstream KeyErrors
    # --------------------------------------------------------

    if len(image_df) == 0:

        raise RuntimeError(
            "No images could be successfully read. "
            "Check paths, permissions, image formats "
            "or dataset corruption."
        )

    if len(instance_df) == 0:

        raise RuntimeError(
            "Images were read successfully, but no "
            "valid segmentation instances survived "
            "rasterization."
        )

    # ========================================================
    # SAVE AUDIT TABLES
    # ========================================================

    image_df.to_csv(
        audit_dir
        / "images.csv",
        index=False,
    )

    instance_df.to_csv(
        audit_dir
        / "instances.csv",
        index=False,
    )

    # ========================================================
    # RESOLUTION DISTRIBUTION
    # ========================================================

    resolution_df = (
        image_df
        .groupby(
            [
                "width",
                "height",
            ]
        )
        .size()
        .reset_index(
            name="image_count"
        )
        .sort_values(
            "image_count",
            ascending=False,
        )
    )

    resolution_df[
        "fraction"
    ] = (
        resolution_df[
            "image_count"
        ]
        / len(image_df)
    )

    resolution_df.to_csv(
        audit_dir
        / "resolution_distribution.csv",
        index=False,
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    class_distribution = (
        instance_df[
            "class_id"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "class_id"
        )
        .reset_index(
            name="instance_count"
        )
    )

    class_distribution.to_csv(
        audit_dir
        / "class_distribution.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        # ----------------------------------------------------
        # Dataset size
        # ----------------------------------------------------

        "total_images_found":
            int(len(pairs)),

        "total_images_readable":
            int(len(image_df)),

        "missing_label_images":
            int(
                len(
                    missing_labels
                )
            ),

        "unreadable_images":
            int(
                len(
                    unreadable_images
                )
            ),

        "total_valid_instances":
            int(
                len(
                    instance_df
                )
            ),

        "total_degenerate_instances":
            int(
                total_degenerate_instances
            ),

        # ----------------------------------------------------
        # Split counts
        # ----------------------------------------------------

        "train_images":
            int(
                (
                    image_df[
                        "split"
                    ]
                    == "train"
                ).sum()
            ),

        "val_images":
            int(
                (
                    image_df[
                        "split"
                    ]
                    == "val"
                ).sum()
            ),

        "test_images":
            int(
                (
                    image_df[
                        "split"
                    ]
                    == "test"
                ).sum()
            ),

        # ----------------------------------------------------
        # Parser stats
        # ----------------------------------------------------

        "annotation_total_lines":
            int(
                dataset_parser_stats[
                    "total_lines"
                ]
            ),

        "annotation_empty_lines":
            int(
                dataset_parser_stats[
                    "empty_lines"
                ]
            ),

        "annotation_malformed_lines":
            int(
                dataset_parser_stats[
                    "malformed_lines"
                ]
            ),

        "annotation_out_of_range_polygons":
            int(
                dataset_parser_stats[
                    "out_of_range_polygons"
                ]
            ),

        "annotation_valid_polygons":
            int(
                dataset_parser_stats[
                    "valid_polygons"
                ]
            ),

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        "min_image_width":
            int(
                image_df[
                    "width"
                ].min()
            ),

        "max_image_width":
            int(
                image_df[
                    "width"
                ].max()
            ),

        "median_image_width":
            float(
                image_df[
                    "width"
                ].median()
            ),

        "min_image_height":
            int(
                image_df[
                    "height"
                ].min()
            ),

        "max_image_height":
            int(
                image_df[
                    "height"
                ].max()
            ),

        "median_image_height":
            float(
                image_df[
                    "height"
                ].median()
            ),

        "unique_resolutions":
            int(
                image_df[
                    [
                        "width",
                        "height",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        # ----------------------------------------------------
        # Instances per image
        # ----------------------------------------------------

        "mean_instances_per_image":
            float(
                image_df[
                    "valid_instances"
                ].mean()
            ),

        "median_instances_per_image":
            float(
                image_df[
                    "valid_instances"
                ].median()
            ),

        # ----------------------------------------------------
        # Instance geometry
        # ----------------------------------------------------

        "mean_mask_area_pixels":
            float(
                instance_df[
                    "area_pixels"
                ].mean()
            ),

        "median_mask_area_pixels":
            float(
                instance_df[
                    "area_pixels"
                ].median()
            ),

        "mean_max_local_width_px":
            float(
                instance_df[
                    "max_local_width_px"
                ].mean()
            ),

        "median_max_local_width_px":
            float(
                instance_df[
                    "max_local_width_px"
                ].median()
            ),

        "border_touching_instances":
            int(
                instance_df[
                    "touches_border"
                ].sum()
            ),

        "border_touching_fraction":
            float(
                instance_df[
                    "touches_border"
                ].mean()
            ),

        # ----------------------------------------------------
        # Foreground
        # ----------------------------------------------------

        "total_foreground_pixels":
            int(
                image_df[
                    "foreground_pixels"
                ].sum()
            ),

        "mean_foreground_fraction":
            float(
                image_df[
                    "foreground_fraction"
                ].mean()
            ),

        "median_foreground_fraction":
            float(
                image_df[
                    "foreground_fraction"
                ].median()
            ),

        # ----------------------------------------------------
        # Overlap
        # ----------------------------------------------------

        "total_overlap_pixels":
            int(
                image_df[
                    "overlap_pixels"
                ].sum()
            ),

        "images_with_overlap":
            int(
                (
                    image_df[
                        "overlap_pixels"
                    ]
                    > 0
                ).sum()
            ),

        "fraction_images_with_overlap":
            float(
                (
                    image_df[
                        "overlap_pixels"
                    ]
                    > 0
                ).mean()
            ),
    }

    # --------------------------------------------------------
    # Dataset-wide overlap fraction
    # --------------------------------------------------------

    total_fg = int(
        image_df[
            "foreground_pixels"
        ].sum()
    )

    total_overlap = int(
        image_df[
            "overlap_pixels"
        ].sum()
    )

    summary[
        "dataset_overlap_fraction_foreground"
    ] = (
        float(
            total_overlap
            / total_fg
        )
        if total_fg > 0
        else 0.0
    )

    # ========================================================
    # ADD PERCENTILES
    # ========================================================

    add_percentiles(
        summary,
        instance_df,
    )

    # ========================================================
    # ADD THIN-INSTANCE COUNTS
    # ========================================================

    add_thin_instance_statistics(
        summary,
        instance_df,
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        audit_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # ========================================================
    # RANDOM PREVIEW GENERATION
    # ========================================================

    num_previews = min(
        args.num_previews,
        len(pairs),
    )

    preview_rng = random.Random(
        args.seed
    )

    preview_pairs = (
        preview_rng.sample(
            pairs,
            num_previews,
        )
    )

    print()
    print("=" * 80)
    print("GENERATING PREVIEWS")
    print("=" * 80)

    successful_previews = 0

    for idx, pair in enumerate(
        preview_pairs,
        start=1,
    ):

        output_path = (
            preview_dir
            / (
                f"{pair['image_path'].stem}"
                "_preview.jpg"
            )
        )

        success = visualize_image(
            pair[
                "image_path"
            ],
            pair[
                "label_path"
            ],
            output_path,
        )

        if success:

            successful_previews += 1

            print(
                f"[{idx}/{num_previews}] "
                f"{output_path.name}"
            )

    # ========================================================
    # FINAL TERMINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    important_keys = [
        "total_images_found",
        "total_images_readable",
        "missing_label_images",
        "unreadable_images",
        "total_valid_instances",
        "total_degenerate_instances",

        "train_images",
        "val_images",
        "test_images",

        "min_image_width",
        "max_image_width",
        "min_image_height",
        "max_image_height",
        "unique_resolutions",

        "mean_instances_per_image",
        "median_instances_per_image",

        "mean_mask_area_pixels",
        "median_mask_area_pixels",

        "mean_max_local_width_px",
        "median_max_local_width_px",

        "max_local_width_p1",
        "max_local_width_p5",
        "max_local_width_p10",
        "max_local_width_p25",
        "max_local_width_p50",
        "max_local_width_p90",
        "max_local_width_p95",

        "fraction_width_le_2px",
        "fraction_width_le_3px",
        "fraction_width_le_5px",
        "fraction_width_le_8px",

        "median_foreground_fraction",

        "images_with_overlap",
        "fraction_images_with_overlap",
        "dataset_overlap_fraction_foreground",

        "border_touching_instances",

        "annotation_malformed_lines",
        "annotation_out_of_range_polygons",
    ]

    for key in important_keys:

        if key in summary:

            print(
                f"{key:42s}: "
                f"{summary[key]}"
            )

    print()
    print(
        f"Successful previews: "
        f"{successful_previews}/"
        f"{num_previews}"
    )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"Results directory:\n"
        f"  {output_dir}"
    )

    print()
    print("Split files:")

    print(
        f"  {splits_dir / 'train.txt'}"
    )

    print(
        f"  {splits_dir / 'val.txt'}"
    )

    print(
        f"  {splits_dir / 'test.txt'}"
    )

    print()
    print("Audit files:")

    print(
        f"  {audit_dir / 'images.csv'}"
    )

    print(
        f"  {audit_dir / 'instances.csv'}"
    )

    print(
        f"  {audit_dir / 'summary.json'}"
    )

    print(
        f"  {audit_dir / 'resolution_distribution.csv'}"
    )

    print(
        f"  {audit_dir / 'class_distribution.csv'}"
    )

    print(
        f"  {audit_dir / 'annotation_issues.csv'}"
    )

    print(
        f"  {audit_dir / 'missing_labels.txt'}"
    )

    print(
        f"  {audit_dir / 'unreadable_images.txt'}"
    )

    print()
    print(
        f"Preview directory:\n"
        f"  {preview_dir}"
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare and audit a YOLO segmentation "
            "dataset for hair-shaft SAM fine-tuning."
        )
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help=(
            "Directory containing dataset images."
        ),
    )

    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help=(
            "Directory containing YOLO polygon labels."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="hair_sam_preparation",
        help=(
            "Directory where audit results are saved."
        ),
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--num_previews",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--group_regex",
        type=str,
        default=None,
        help=(
            "Optional regex used to extract patient/session "
            "group from image filenames. Images belonging "
            "to the same group remain in the same split. "
            "Example: '^(patient[0-9]+)'"
        ),
    )

    args = parser.parse_args()

    main(args)