import os
import cv2
import random
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing images
IMAGE_DIR = r"C:\Users\DELL\Desktop\Hairs\segmentation\dataset\synthetic_v3\synthetic_v3\images"

# Folder containing YOLO segmentation labels
LABEL_DIR = r"C:\Users\DELL\Desktop\Hairs\segmentation\dataset\synthetic_v3\synthetic_v3\labels"

# Folder where visualization results will be saved
OUTPUT_DIR = r"C:\Users\DELL\Desktop\Hairs\segmentation\dataset\synthetic_v3\example_visualizations"

# Number of random images to visualize
NUM_IMAGES = 50

# Transparency of segmentation mask
MASK_ALPHA = 0.40

# Random seed
# Set to None if you want different random images every run
RANDOM_SEED = 42


# ============================================================
# CLASS COLORS
# ============================================================

# BGR colors used by OpenCV.
# Add/change colors if you have multiple classes.
CLASS_COLORS = [
    (0, 0, 255),       # Class 0 - Red
    (0, 255, 0),       # Class 1 - Green
    (255, 0, 0),       # Class 2 - Blue
    (0, 255, 255),     # Class 3 - Yellow
    (255, 0, 255),     # Class 4 - Magenta
    (255, 255, 0),     # Class 5 - Cyan
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (128, 255, 0),
]


# ============================================================
# READ YOLO SEGMENTATION LABEL
# ============================================================

def read_yolo_segmentation(label_path, image_width, image_height):
    """
    Reads YOLO segmentation annotations.

    Format:
        class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates are normalized between 0 and 1.

    Returns:
        [
            {
                "class_id": int,
                "polygon": np.ndarray of shape (N, 2)
            },
            ...
        ]
    """

    annotations = []

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            # First value = class ID
            class_id = int(float(values[0]))

            # Remaining values = polygon coordinates
            coords = list(map(float, values[1:]))

            # Polygon requires x, y pairs
            if len(coords) < 6 or len(coords) % 2 != 0:
                print(f"Warning: Invalid polygon in {label_path}")
                continue

            points = []

            for i in range(0, len(coords), 2):

                x_norm = coords[i]
                y_norm = coords[i + 1]

                # Convert normalized coordinates to pixel coordinates
                x = int(round(x_norm * image_width))
                y = int(round(y_norm * image_height))

                # Keep coordinates inside image
                x = np.clip(x, 0, image_width - 1)
                y = np.clip(y, 0, image_height - 1)

                points.append([x, y])

            polygon = np.array(points, dtype=np.int32)

            annotations.append(
                {
                    "class_id": class_id,
                    "polygon": polygon
                }
            )

    return annotations


# ============================================================
# DRAW SEGMENTATION MASK
# ============================================================

def draw_segmentation(image, annotations, alpha=0.4):
    """
    Draw polygon masks and polygon boundaries on image.
    """

    # Copy for transparent filled masks
    overlay = image.copy()

    # --------------------------------------------------------
    # Draw filled masks
    # --------------------------------------------------------

    for annotation in annotations:

        class_id = annotation["class_id"]
        polygon = annotation["polygon"]

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        cv2.fillPoly(
            overlay,
            [polygon],
            color
        )

    # Blend mask with original image
    visualized = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0
    )

    # --------------------------------------------------------
    # Draw polygon boundaries
    # --------------------------------------------------------

    for annotation in annotations:

        class_id = annotation["class_id"]
        polygon = annotation["polygon"]

        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        cv2.polylines(
            visualized,
            [polygon],
            isClosed=True,
            color=color,
            thickness=2
        )

    return visualized


# ============================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================

def visualize_random_segmentation(
    image_dir,
    label_dir,
    output_dir,
    num_images=20,
    alpha=0.4,
    random_seed=42
):

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # Create result folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    # --------------------------------------------------------
    # Find images that have corresponding labels
    # --------------------------------------------------------

    valid_images = []

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in image_extensions:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            valid_images.append(image_path)

    print(f"Images with labels found: {len(valid_images)}")

    if len(valid_images) == 0:
        raise RuntimeError(
            "No image-label pairs were found. "
            "Check IMAGE_DIR and LABEL_DIR."
        )

    # --------------------------------------------------------
    # Select random images
    # --------------------------------------------------------

    if random_seed is not None:
        random.seed(random_seed)

    num_images = min(num_images, len(valid_images))

    selected_images = random.sample(
        valid_images,
        num_images
    )

    print(f"Randomly selected images: {num_images}")

    # --------------------------------------------------------
    # Process selected images
    # --------------------------------------------------------

    for idx, image_path in enumerate(selected_images, start=1):

        label_path = label_dir / f"{image_path.stem}.txt"

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]

        # Read segmentation polygons
        annotations = read_yolo_segmentation(
            label_path,
            width,
            height
        )

        # Draw masks
        visualized = draw_segmentation(
            image,
            annotations,
            alpha=alpha
        )

        # ----------------------------------------------------
        # Add simple information to image
        # ----------------------------------------------------

        text = f"Masks: {len(annotations)}"

        cv2.putText(
            visualized,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

        output_path = output_dir / f"{image_path.stem}_mask.jpg"

        cv2.imwrite(
            str(output_path),
            visualized
        )

        print(
            f"[{idx}/{num_images}] "
            f"{image_path.name} -> "
            f"{output_path.name} "
            f"({len(annotations)} masks)"
        )

    print("\nVisualization completed.")
    print(f"Results saved in: {output_dir}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    visualize_random_segmentation(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        num_images=NUM_IMAGES,
        alpha=MASK_ALPHA,
        random_seed=RANDOM_SEED
    )