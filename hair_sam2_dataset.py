import cv2
import random
import linecache
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# ============================================================
# READ A SINGLE YOLO POLYGON
# ============================================================

def read_yolo_polygon(
    label_path,
    line_number,
    image_width,
    image_height,
):
    """
    Read exactly one polygon from the YOLO segmentation file.

    line_number is 1-based because it comes from
    source_label_line in instances.csv.

    Format:
        class_id x1 y1 x2 y2 ...
    """

    line = linecache.getline(
        str(label_path),
        int(line_number),
    ).strip()

    if not line:
        raise RuntimeError(
            f"Could not read line {line_number} "
            f"from {label_path}"
        )

    values = line.split()

    coords = np.asarray(
        list(map(float, values[1:])),
        dtype=np.float32,
    ).reshape(-1, 2)

    coords = np.clip(
        coords,
        0.0,
        1.0,
    )

    # Same coordinate convention as the audit code
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

    return polygon


# ============================================================
# GET BOUNDING BOX FROM MASK
# ============================================================

def get_bbox_from_mask(mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return np.array(
        [
            xs.min(),
            ys.min(),
            xs.max(),
            ys.max(),
        ],
        dtype=np.float32,
    )


# ============================================================
# CHOOSE CROP START
# ============================================================

def choose_crop_start(
    object_min,
    object_max,
    dimension,
    crop_size,
    training,
):
    """
    Pick a crop position that contains the entire target
    whenever possible.

    When the instance is larger than the crop, we crop
    around the object's center with some random jitter.
    """

    max_start = dimension - crop_size

    if max_start <= 0:
        return 0

    object_size = (
        object_max
        - object_min
        + 1
    )

    # --------------------------------------------------------
    # Case 1:
    # Target can fit completely inside crop
    # --------------------------------------------------------

    if object_size <= crop_size:

        minimum_start = max(
            0,
            int(object_max - crop_size + 1),
        )

        maximum_start = min(
            int(object_min),
            max_start,
        )

        if minimum_start <= maximum_start:

            if training:

                return random.randint(
                    minimum_start,
                    maximum_start,
                )

            else:

                return (
                    minimum_start
                    + maximum_start
                ) // 2

    # --------------------------------------------------------
    # Case 2:
    # Target itself is larger than crop
    # --------------------------------------------------------

    center = (
        object_min
        + object_max
    ) / 2.0

    start = int(
        round(
            center
            - crop_size / 2
        )
    )

    if training:

        jitter = int(
            crop_size * 0.15
        )

        start += random.randint(
            -jitter,
            jitter,
        )

    start = np.clip(
        start,
        0,
        max_start,
    )

    return int(start)


# ============================================================
# PAD SMALL IMAGE
# ============================================================

def pad_to_crop_size(
    image,
    polygon,
    crop_size,
    training,
):
    """
    If image is smaller than 1024 in either dimension,
    pad instead of forcibly upscaling it.

    Padding position is randomized during training so the
    model doesn't learn that padded images always start
    in the upper-left corner.
    """

    h, w = image.shape[:2]

    required_w = max(
        0,
        crop_size - w,
    )

    required_h = max(
        0,
        crop_size - h,
    )

    if required_w == 0 and required_h == 0:

        return image, polygon

    if training:

        pad_left = random.randint(
            0,
            required_w,
        )

        pad_top = random.randint(
            0,
            required_h,
        )

    else:

        pad_left = (
            required_w // 2
        )

        pad_top = (
            required_h // 2
        )

    pad_right = (
        required_w - pad_left
    )

    pad_bottom = (
        required_h - pad_top
    )

    # Use image median as padding value rather than
    # introducing a large artificial black region.
    median_value = np.median(
        image.reshape(-1, 3),
        axis=0,
    )

    median_value = tuple(
        int(x)
        for x in median_value
    )

    image = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=median_value,
    )

    polygon = polygon.copy()

    polygon[:, 0] += pad_left
    polygon[:, 1] += pad_top

    return image, polygon


# ============================================================
# DATASET
# ============================================================

class HairSAM2Dataset(Dataset):

    def __init__(
        self,
        images_csv,
        instances_csv,
        split="train",
        crop_size=1024,
        scale_min=0.85,
        scale_max=1.15,
        box_jitter=20,
        max_instances=None,
        seed=42,
        training=True,
    ):

        self.crop_size = crop_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.box_jitter = box_jitter
        self.training = training

        # ----------------------------------------------------
        # Load audit output
        # ----------------------------------------------------

        images_df = pd.read_csv(
            images_csv
        )

        instances_df = pd.read_csv(
            instances_csv
        )

        # ----------------------------------------------------
        # Add label path to every instance
        # ----------------------------------------------------

        label_lookup = (
            images_df
            .set_index("image_path")
            ["label_path"]
            .to_dict()
        )

        instances_df[
            "label_path"
        ] = (
            instances_df[
                "image_path"
            ]
            .map(label_lookup)
        )

        # ----------------------------------------------------
        # Filter desired split
        # ----------------------------------------------------

        instances_df = (
            instances_df[
                instances_df["split"]
                == split
            ]
            .copy()
        )

        instances_df = (
            instances_df
            .dropna(
                subset=[
                    "label_path"
                ]
            )
        )

        # ----------------------------------------------------
        # Optional subset for fast experiments
        # ----------------------------------------------------

        if (
            max_instances is not None
            and max_instances
            < len(instances_df)
        ):

            instances_df = (
                instances_df.sample(
                    n=max_instances,
                    random_state=seed,
                )
            )

        self.df = (
            instances_df
            .reset_index(
                drop=True
            )
        )

        print(
            f"{split} dataset: "
            f"{len(self.df):,} instances"
        )


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = cv2.imread(
            str(
                row["image_path"]
            )
        )

        if image is None:

            raise RuntimeError(
                "Could not read image: "
                f"{row['image_path']}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_h, original_w = (
            image.shape[:2]
        )

        # ====================================================
        # LOAD TARGET POLYGON
        # ====================================================

        polygon = read_yolo_polygon(
            label_path=row[
                "label_path"
            ],
            line_number=row[
                "source_label_line"
            ],
            image_width=original_w,
            image_height=original_h,
        )

        # ====================================================
        # RANDOM SCALE AUGMENTATION
        # ====================================================

        if self.training:

            scale = random.uniform(
                self.scale_min,
                self.scale_max,
            )

        else:

            scale = 1.0

        new_w = max(
            1,
            int(
                round(
                    original_w
                    * scale
                )
            ),
        )

        new_h = max(
            1,
            int(
                round(
                    original_h
                    * scale
                )
            ),
        )

        # Actual scaling may differ microscopically because
        # image sizes must be integers.
        scale_x = (
            new_w / original_w
        )

        scale_y = (
            new_h / original_h
        )

        if (
            new_w != original_w
            or new_h != original_h
        ):

            image = cv2.resize(
                image,
                (
                    new_w,
                    new_h,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            polygon = (
                polygon
                .astype(np.float32)
            )

            polygon[:, 0] *= scale_x
            polygon[:, 1] *= scale_y

            polygon = np.round(
                polygon
            ).astype(
                np.int32
            )

        # ====================================================
        # PAD IF IMAGE < 1024
        # ====================================================

        image, polygon = (
            pad_to_crop_size(
                image=image,
                polygon=polygon,
                crop_size=self.crop_size,
                training=self.training,
            )
        )

        h, w = image.shape[:2]

        # ====================================================
        # TARGET BBOX BEFORE CROP
        # ====================================================

        px_min = int(
            polygon[:, 0].min()
        )

        px_max = int(
            polygon[:, 0].max()
        )

        py_min = int(
            polygon[:, 1].min()
        )

        py_max = int(
            polygon[:, 1].max()
        )

        # ====================================================
        # INSTANCE-CENTERED RANDOM CROP
        # ====================================================

        crop_x = choose_crop_start(
            object_min=px_min,
            object_max=px_max,
            dimension=w,
            crop_size=self.crop_size,
            training=self.training,
        )

        crop_y = choose_crop_start(
            object_min=py_min,
            object_max=py_max,
            dimension=h,
            crop_size=self.crop_size,
            training=self.training,
        )

        image = image[
            crop_y:
            crop_y + self.crop_size,

            crop_x:
            crop_x + self.crop_size,
        ]

        local_polygon = (
            polygon.copy()
        )

        local_polygon[:, 0] -= (
            crop_x
        )

        local_polygon[:, 1] -= (
            crop_y
        )

        # ====================================================
        # RASTERIZE TARGET ONLY AFTER CROPPING
        # ====================================================

        mask = np.zeros(
            (
                self.crop_size,
                self.crop_size,
            ),
            dtype=np.uint8,
        )

        # OpenCV clips polygon geometry at image boundaries,
        # so vertices are intentionally NOT individually
        # clipped beforehand.
        cv2.fillPoly(
            mask,
            [local_polygon],
            1,
        )

        if mask.sum() == 0:

            raise RuntimeError(
                "Target vanished after crop:\n"
                f"{row['image_path']}\n"
                f"instance={row['instance_id']}"
            )

        # ====================================================
        # SIMPLE GEOMETRIC AUGMENTATION
        # ====================================================

        if (
            self.training
            and random.random() < 0.5
        ):

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            mask = np.ascontiguousarray(
                mask[:, ::-1]
            )

        if (
            self.training
            and random.random() < 0.5
        ):

            image = np.ascontiguousarray(
                image[::-1, :]
            )

            mask = np.ascontiguousarray(
                mask[::-1, :]
            )

        # ====================================================
        # MILD PHOTOMETRIC AUGMENTATION
        # ====================================================

        if self.training:

            contrast = random.uniform(
                0.90,
                1.10,
            )

            brightness = random.uniform(
                -10.0,
                10.0,
            )

            image = (
                image.astype(
                    np.float32
                )
                * contrast
                + brightness
            )

            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        # ====================================================
        # BOX PROMPT FROM FINAL MASK
        # ====================================================

        bbox = get_bbox_from_mask(
            mask
        )

        if bbox is None:

            raise RuntimeError(
                "Final target mask is empty."
            )

        x1, y1, x2, y2 = bbox

        # ----------------------------------------------------
        # Outward box jitter
        #
        # Keeps the target inside the prompt while teaching
        # SAM not to depend on perfectly tight GT boxes.
        # ----------------------------------------------------

        if self.training:

            x1 -= random.randint(
                0,
                self.box_jitter,
            )

            y1 -= random.randint(
                0,
                self.box_jitter,
            )

            x2 += random.randint(
                0,
                self.box_jitter,
            )

            y2 += random.randint(
                0,
                self.box_jitter,
            )

        x1 = np.clip(
            x1,
            0,
            self.crop_size - 1,
        )

        y1 = np.clip(
            y1,
            0,
            self.crop_size - 1,
        )

        x2 = np.clip(
            x2,
            0,
            self.crop_size - 1,
        )

        y2 = np.clip(
            y2,
            0,
            self.crop_size - 1,
        )

        bbox = np.asarray(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=np.float32,
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        return {
            "image":
                image,

            "mask":
                torch.from_numpy(
                    mask
                )
                .float()
                .unsqueeze(0),

            "box":
                torch.from_numpy(
                    bbox
                ),

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
        }