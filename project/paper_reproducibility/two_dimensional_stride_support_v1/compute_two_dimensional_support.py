from __future__ import annotations

import csv
import glob
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from PIL import Image, ImageFile


# ============================================================
# Fixed paths and statistical protocol
# ============================================================

ROOT = Path(r"D:\YOLO-WTB")
DATA_YAML = (
    ROOT
    / "data"
    / "wtb_damage_4021_1151.yaml"
)

WORK = (
    ROOT
    / "paper_reproducibility"
    / "two_dimensional_stride_support_v1"
)

REPORTS = WORK / "reports"
RAW = WORK / "raw_statistics"

REPORTS.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 960

LEVELS = [
    ("P2", 4),
    ("P3", 8),
    ("P4", 16),
    ("P5", 32),
]

EXPECTED_TRAIN_IMAGES = 4021
EXPECTED_VAL_IMAGES = 1151
EXPECTED_DEVELOPMENT_INSTANCES = 21834

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# ============================================================
# Console and formatting utilities
# ============================================================

def stage(message: str) -> None:
    print("\n" + "=" * 88, flush=True)
    print(message, flush=True)
    print("=" * 88, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"

    return f"{minutes:02d}:{remaining:02d}"


def percentage(count: int, total: int) -> float:
    if total <= 0:
        return float("nan")

    return 100.0 * count / total


def latex_percentage(count: int, total: int) -> str:
    return f"{percentage(count, total):.2f}\\%"


def path_key(path: Path) -> str:
    return str(path.resolve()).lower()


# ============================================================
# Dataset YAML and image-list resolution
# ============================================================

def load_dataset_yaml(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Dataset YAML is not a dictionary: {path}"
        )

    if "train" not in data:
        raise KeyError(
            f"Dataset YAML has no train entry: {path}"
        )

    if "val" not in data:
        raise KeyError(
            f"Dataset YAML has no val entry: {path}"
        )

    return data


def first_existing(
    candidates: Iterable[Path],
) -> Path:
    candidate_list = []

    for candidate in candidates:
        resolved = candidate.expanduser()

        try:
            resolved = resolved.resolve()
        except Exception:
            pass

        candidate_list.append(resolved)

        if resolved.exists():
            return resolved

    return candidate_list[0]


def resolve_dataset_root(
    config: Dict[str, Any],
    yaml_path: Path,
) -> Path:
    yaml_directory = yaml_path.parent.resolve()
    raw_root = config.get("path")

    if raw_root is None or str(raw_root).strip() == "":
        return yaml_directory

    root_path = Path(str(raw_root))

    if root_path.is_absolute():
        return root_path.resolve()

    return first_existing(
        [
            yaml_directory / root_path,
            ROOT / root_path,
            Path.cwd() / root_path,
        ]
    )


def resolve_source_path(
    value: str,
    dataset_root: Path,
    yaml_directory: Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    return first_existing(
        [
            dataset_root / path,
            yaml_directory / path,
            ROOT / path,
        ]
    )


def resolve_txt_image_path(
    value: str,
    txt_path: Path,
    dataset_root: Path,
    yaml_directory: Path,
) -> Path:
    text = value.strip().strip('"').strip("'")

    if text.startswith("./") or text.startswith(".\\"):
        text = text[2:]

    path = Path(text)

    if path.is_absolute():
        return path.resolve()

    return first_existing(
        [
            txt_path.parent / path,
            dataset_root / path,
            yaml_directory / path,
            ROOT / path,
        ]
    )


def collect_from_path(
    source_path: Path,
    dataset_root: Path,
    yaml_directory: Path,
) -> List[Path]:
    images: List[Path] = []

    source_text = str(source_path)

    if any(
        token in source_text
        for token in ("*", "?", "[")
    ):
        for item in glob.glob(
            source_text,
            recursive=True,
        ):
            item_path = Path(item)

            if (
                item_path.is_file()
                and item_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                images.append(item_path.resolve())

        return images

    if source_path.is_dir():
        for item in source_path.rglob("*"):
            if (
                item.is_file()
                and item.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                images.append(item.resolve())

        return images

    if source_path.is_file():
        if (
            source_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            return [source_path.resolve()]

        if source_path.suffix.lower() == ".txt":
            with source_path.open(
                "r",
                encoding="utf-8-sig",
            ) as file:
                for line in file:
                    text = line.strip()

                    if (
                        not text
                        or text.startswith("#")
                    ):
                        continue

                    image_path = (
                        resolve_txt_image_path(
                            text,
                            source_path,
                            dataset_root,
                            yaml_directory,
                        )
                    )

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    ):
                        images.append(
                            image_path.resolve()
                        )
                    else:
                        raise FileNotFoundError(
                            "Image listed in dataset TXT "
                            f"was not found:\n{text}\n"
                            f"Resolved as:\n{image_path}"
                        )

            return images

    raise FileNotFoundError(
        f"Dataset source does not exist: {source_path}"
    )


def collect_split_images(
    split_value: Any,
    dataset_root: Path,
    yaml_directory: Path,
) -> List[Path]:
    entries = (
        split_value
        if isinstance(split_value, list)
        else [split_value]
    )

    collected: List[Path] = []

    for entry in entries:
        source_path = resolve_source_path(
            str(entry),
            dataset_root,
            yaml_directory,
        )

        collected.extend(
            collect_from_path(
                source_path,
                dataset_root,
                yaml_directory,
            )
        )

    unique_images: List[Path] = []
    seen = set()

    for image_path in collected:
        key = path_key(image_path)

        if key in seen:
            continue

        seen.add(key)
        unique_images.append(image_path)

    unique_images.sort(
        key=lambda path: str(path).lower()
    )

    return unique_images


# ============================================================
# Image and label handling
# ============================================================

def exif_corrected_size(
    image_path: Path,
) -> Tuple[int, int]:
    with Image.open(str(image_path)) as image:
        width, height = image.size

        try:
            orientation = image.getexif().get(274)
        except Exception:
            orientation = None

        if orientation in {5, 6, 7, 8}:
            width, height = height, width

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: {image_path}"
        )

    return int(width), int(height)


def label_path_from_image(
    image_path: Path,
) -> Path:
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, part in enumerate(parts)
        if part.lower() == "images"
    ]

    if image_indices:
        index = image_indices[-1]
        parts[index] = "labels"

        return Path(*parts).with_suffix(".txt")

    candidates = [
        image_path.with_suffix(".txt"),
        (
            image_path.parent.parent
            / "labels"
            / image_path.parent.name
            / f"{image_path.stem}.txt"
        ),
        (
            image_path.parent.parent
            / "labels"
            / f"{image_path.stem}.txt"
        ),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def polygon_to_box(
    coordinates: List[float],
) -> Tuple[float, float, float, float]:
    xs = coordinates[0::2]
    ys = coordinates[1::2]

    minimum_x = min(xs)
    maximum_x = max(xs)
    minimum_y = min(ys)
    maximum_y = max(ys)

    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    center_x = (minimum_x + maximum_x) / 2.0
    center_y = (minimum_y + maximum_y) / 2.0

    return center_x, center_y, width, height


def parse_label_file(
    label_path: Path,
) -> Tuple[
    List[Dict[str, float]],
    int,
    int,
    List[Dict[str, Any]],
]:
    if not label_path.is_file():
        return [], 0, 0, []

    with label_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        lines = file.readlines()

    annotations: List[Dict[str, float]] = []
    duplicate_count = 0
    segment_count = 0
    errors: List[Dict[str, Any]] = []
    seen = set()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        text = raw_line.strip()

        if not text:
            continue

        try:
            values = [
                float(value)
                for value in text.split()
            ]
        except Exception as exception:
            errors.append(
                {
                    "label_path": str(label_path),
                    "line_number": line_number,
                    "line": text,
                    "error": (
                        "Non-numeric label value: "
                        f"{exception}"
                    ),
                }
            )
            continue

        if len(values) == 5:
            class_id, center_x, center_y, width, height = (
                values
            )

        elif (
            len(values) > 5
            and (len(values) - 1) % 2 == 0
        ):
            class_id = values[0]
            coordinates = values[1:]

            (
                center_x,
                center_y,
                width,
                height,
            ) = polygon_to_box(coordinates)

            segment_count += 1

        else:
            errors.append(
                {
                    "label_path": str(label_path),
                    "line_number": line_number,
                    "line": text,
                    "error": (
                        "Expected YOLO box format with "
                        "5 values or polygon format with "
                        "class plus coordinate pairs."
                    ),
                }
            )
            continue

        numeric_values = [
            class_id,
            center_x,
            center_y,
            width,
            height,
        ]

        if not all(
            math.isfinite(value)
            for value in numeric_values
        ):
            errors.append(
                {
                    "label_path": str(label_path),
                    "line_number": line_number,
                    "line": text,
                    "error": "Non-finite label value.",
                }
            )
            continue

        tolerance = 1.0e-3

        valid = all(
            [
                class_id >= 0,
                abs(class_id - round(class_id))
                <= 1.0e-6,
                width > 0,
                height > 0,
                center_x >= -tolerance,
                center_x <= 1.0 + tolerance,
                center_y >= -tolerance,
                center_y <= 1.0 + tolerance,
                width <= 1.0 + tolerance,
                height <= 1.0 + tolerance,
                center_x - width / 2.0
                >= -tolerance,
                center_x + width / 2.0
                <= 1.0 + tolerance,
                center_y - height / 2.0
                >= -tolerance,
                center_y + height / 2.0
                <= 1.0 + tolerance,
            ]
        )

        if not valid:
            errors.append(
                {
                    "label_path": str(label_path),
                    "line_number": line_number,
                    "line": text,
                    "error": (
                        "Invalid class or normalized "
                        "bounding-box coordinates."
                    ),
                }
            )
            continue

        unique_key = (
            int(round(class_id)),
            round(center_x, 12),
            round(center_y, 12),
            round(width, 12),
            round(height, 12),
        )

        if unique_key in seen:
            duplicate_count += 1
            continue

        seen.add(unique_key)

        annotations.append(
            {
                "class_id": int(round(class_id)),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width_norm": float(width),
                "height_norm": float(height),
                "line_number": line_number,
            }
        )

    return (
        annotations,
        duplicate_count,
        segment_count,
        errors,
    )


# ============================================================
# Main processing
# ============================================================

stage("[1/6] Loading WTMD development split")

config = load_dataset_yaml(DATA_YAML)
yaml_directory = DATA_YAML.parent.resolve()

dataset_root = resolve_dataset_root(
    config,
    DATA_YAML,
)

print(f"Dataset YAML : {DATA_YAML}", flush=True)
print(f"Dataset root : {dataset_root}", flush=True)
print(f"Target size  : {TARGET_SIZE} x {TARGET_SIZE}", flush=True)
print("Included     : train + val", flush=True)
print("Excluded     : test", flush=True)


stage("[2/6] Resolving train and validation images")

train_images = collect_split_images(
    config["train"],
    dataset_root,
    yaml_directory,
)

val_images = collect_split_images(
    config["val"],
    dataset_root,
    yaml_directory,
)

train_keys = {
    path_key(path)
    for path in train_images
}

val_keys = {
    path_key(path)
    for path in val_images
}

overlap_keys = train_keys.intersection(val_keys)

print(
    f"Train images : {len(train_images)} "
    f"(expected {EXPECTED_TRAIN_IMAGES})",
    flush=True,
)

print(
    f"Val images   : {len(val_images)} "
    f"(expected {EXPECTED_VAL_IMAGES})",
    flush=True,
)

print(
    f"Path overlap : {len(overlap_keys)}",
    flush=True,
)

if overlap_keys:
    overlap_path = (
        REPORTS
        / "train_val_path_overlap.txt"
    )

    overlap_path.write_text(
        "\n".join(sorted(overlap_keys)) + "\n",
        encoding="utf-8",
    )

    raise RuntimeError(
        "Train and val contain overlapping image "
        f"paths. Review:\n{overlap_path}"
    )


stage("[3/6] Computing resized two-dimensional support")

geometry_csv = (
    RAW
    / "instance_geometry_after_960_letterbox.csv"
)

support_csv = (
    RAW
    / "instance_support_by_prediction_level.csv"
)

error_csv = (
    REPORTS
    / "annotation_or_image_errors.csv"
)

geometry_fields = [
    "instance_id",
    "split",
    "image_path",
    "label_path",
    "label_line",
    "class_id",
    "original_width",
    "original_height",
    "letterbox_gain",
    "resized_width",
    "resized_height",
    "resized_long_side",
    "resized_short_side",
    "area_equivalent_size",
    "aspect_ratio",
]

support_fields = [
    "instance_id",
    "split",
    "image_path",
    "label_line",
    "class_id",
    "level",
    "stride",
    "q_width",
    "q_height",
    "q_max",
    "q_min",
    "q_area",
    "q_min_lt_1",
    "q_min_lt_2",
    "q_area_lt_1",
    "q_area_lt_4",
    "q_max_lt_1_and_q_min_lt_1",
    "q_max_ge_1_and_q_min_lt_1",
    "q_max_lt_2_and_q_min_lt_2",
    "q_max_ge_2_and_q_min_lt_2",
]

summary = {
    level: defaultdict(int)
    for level, _ in LEVELS
}

split_image_counts = {
    "train": len(train_images),
    "val": len(val_images),
}

split_instance_counts = {
    "train": 0,
    "val": 0,
}

missing_label_counts = {
    "train": 0,
    "val": 0,
}

empty_label_counts = {
    "train": 0,
    "val": 0,
}

background_image_counts = {
    "train": 0,
    "val": 0,
}

duplicate_label_rows = 0
segment_label_rows = 0
all_errors: List[Dict[str, Any]] = []

instance_id = 0
all_split_images = [
    ("train", path)
    for path in train_images
] + [
    ("val", path)
    for path in val_images
]

total_images = len(all_split_images)
start_time = time.time()

with geometry_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as geometry_file, support_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as support_file:

    geometry_writer = csv.DictWriter(
        geometry_file,
        fieldnames=geometry_fields,
    )

    support_writer = csv.DictWriter(
        support_file,
        fieldnames=support_fields,
    )

    geometry_writer.writeheader()
    support_writer.writeheader()

    for image_index, (
        split_name,
        image_path,
    ) in enumerate(
        all_split_images,
        start=1,
    ):
        label_path = label_path_from_image(
            image_path
        )

        try:
            original_width, original_height = (
                exif_corrected_size(
                    image_path
                )
            )
        except Exception as exception:
            all_errors.append(
                {
                    "label_path": "",
                    "line_number": "",
                    "line": str(image_path),
                    "error": (
                        "Image read failure: "
                        f"{type(exception).__name__}: "
                        f"{exception}"
                    ),
                }
            )
            continue

        if not label_path.is_file():
            missing_label_counts[
                split_name
            ] += 1

        (
            annotations,
            duplicate_count,
            segment_count,
            label_errors,
        ) = parse_label_file(label_path)

        duplicate_label_rows += duplicate_count
        segment_label_rows += segment_count
        all_errors.extend(label_errors)

        if (
            label_path.is_file()
            and len(annotations) == 0
            and not label_errors
        ):
            empty_label_counts[
                split_name
            ] += 1

        if len(annotations) == 0:
            background_image_counts[
                split_name
            ] += 1

        letterbox_gain = min(
            TARGET_SIZE / original_width,
            TARGET_SIZE / original_height,
        )

        for annotation in annotations:
            instance_id += 1
            split_instance_counts[
                split_name
            ] += 1

            resized_width = (
                annotation["width_norm"]
                * original_width
                * letterbox_gain
            )

            resized_height = (
                annotation["height_norm"]
                * original_height
                * letterbox_gain
            )

            resized_long_side = max(
                resized_width,
                resized_height,
            )

            resized_short_side = min(
                resized_width,
                resized_height,
            )

            area_equivalent_size = math.sqrt(
                resized_width
                * resized_height
            )

            aspect_ratio = (
                resized_long_side
                / resized_short_side
            )

            geometry_writer.writerow(
                {
                    "instance_id": instance_id,
                    "split": split_name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_line": (
                        annotation["line_number"]
                    ),
                    "class_id": (
                        annotation["class_id"]
                    ),
                    "original_width": (
                        original_width
                    ),
                    "original_height": (
                        original_height
                    ),
                    "letterbox_gain": (
                        f"{letterbox_gain:.12f}"
                    ),
                    "resized_width": (
                        f"{resized_width:.8f}"
                    ),
                    "resized_height": (
                        f"{resized_height:.8f}"
                    ),
                    "resized_long_side": (
                        f"{resized_long_side:.8f}"
                    ),
                    "resized_short_side": (
                        f"{resized_short_side:.8f}"
                    ),
                    "area_equivalent_size": (
                        f"{area_equivalent_size:.8f}"
                    ),
                    "aspect_ratio": (
                        f"{aspect_ratio:.8f}"
                    ),
                }
            )

            for level, stride in LEVELS:
                q_width = resized_width / stride
                q_height = resized_height / stride

                q_max = max(
                    q_width,
                    q_height,
                )

                q_min = min(
                    q_width,
                    q_height,
                )

                q_area = q_width * q_height

                flags = {
                    "q_min_lt_1": q_min < 1.0,
                    "q_min_lt_2": q_min < 2.0,
                    "q_area_lt_1": q_area < 1.0,
                    "q_area_lt_4": q_area < 4.0,
                    (
                        "q_max_lt_1_and_"
                        "q_min_lt_1"
                    ): (
                        q_max < 1.0
                        and q_min < 1.0
                    ),
                    (
                        "q_max_ge_1_and_"
                        "q_min_lt_1"
                    ): (
                        q_max >= 1.0
                        and q_min < 1.0
                    ),
                    (
                        "q_max_lt_2_and_"
                        "q_min_lt_2"
                    ): (
                        q_max < 2.0
                        and q_min < 2.0
                    ),
                    (
                        "q_max_ge_2_and_"
                        "q_min_lt_2"
                    ): (
                        q_max >= 2.0
                        and q_min < 2.0
                    ),
                }

                summary[level]["total"] += 1

                for flag_name, state in flags.items():
                    if state:
                        summary[level][
                            flag_name
                        ] += 1

                support_writer.writerow(
                    {
                        "instance_id": instance_id,
                        "split": split_name,
                        "image_path": str(
                            image_path
                        ),
                        "label_line": (
                            annotation[
                                "line_number"
                            ]
                        ),
                        "class_id": (
                            annotation[
                                "class_id"
                            ]
                        ),
                        "level": level,
                        "stride": stride,
                        "q_width": (
                            f"{q_width:.8f}"
                        ),
                        "q_height": (
                            f"{q_height:.8f}"
                        ),
                        "q_max": (
                            f"{q_max:.8f}"
                        ),
                        "q_min": (
                            f"{q_min:.8f}"
                        ),
                        "q_area": (
                            f"{q_area:.8f}"
                        ),
                        **{
                            key: int(value)
                            for key, value
                            in flags.items()
                        },
                    }
                )

        if (
            image_index == 1
            or image_index % 50 == 0
            or image_index == total_images
        ):
            elapsed = (
                time.time() - start_time
            )

            rate = (
                image_index / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                (total_images - image_index)
                / rate
                if rate > 0
                else 0.0
            )

            print(
                f"[Progress] "
                f"{image_index}/{total_images} images | "
                f"split={split_name} | "
                f"instances={instance_id} | "
                f"current={image_path.name} | "
                f"elapsed={format_duration(elapsed)} | "
                f"ETA={format_duration(remaining)}",
                flush=True,
            )


# ============================================================
# Data-integrity checks
# ============================================================

stage("[4/6] Verifying annotation integrity")

quality_report = {
    "dataset_yaml": str(DATA_YAML),
    "dataset_root": str(dataset_root),
    "target_size": TARGET_SIZE,
    "included_splits": [
        "train",
        "val",
    ],
    "excluded_splits": [
        "test",
    ],
    "train_images": (
        split_image_counts["train"]
    ),
    "val_images": (
        split_image_counts["val"]
    ),
    "total_development_images": (
        total_images
    ),
    "train_instances": (
        split_instance_counts["train"]
    ),
    "val_instances": (
        split_instance_counts["val"]
    ),
    "total_development_instances": (
        instance_id
    ),
    "expected_development_instances": (
        EXPECTED_DEVELOPMENT_INSTANCES
    ),
    "expected_instance_count_match": (
        instance_id
        == EXPECTED_DEVELOPMENT_INSTANCES
    ),
    "train_missing_label_files": (
        missing_label_counts["train"]
    ),
    "val_missing_label_files": (
        missing_label_counts["val"]
    ),
    "train_empty_label_files": (
        empty_label_counts["train"]
    ),
    "val_empty_label_files": (
        empty_label_counts["val"]
    ),
    "train_background_images": (
        background_image_counts["train"]
    ),
    "val_background_images": (
        background_image_counts["val"]
    ),
    "duplicate_label_rows_removed": (
        duplicate_label_rows
    ),
    "polygon_label_rows_converted": (
        segment_label_rows
    ),
    "annotation_or_image_errors": (
        len(all_errors)
    ),
    "train_val_path_overlap": (
        len(overlap_keys)
    ),
}

quality_path = (
    REPORTS
    / "data_quality_report.json"
)

quality_path.write_text(
    json.dumps(
        quality_report,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    f"Train instances : "
    f"{split_instance_counts['train']}",
    flush=True,
)

print(
    f"Val instances   : "
    f"{split_instance_counts['val']}",
    flush=True,
)

print(
    f"Total instances : {instance_id}",
    flush=True,
)

print(
    f"Expected total  : "
    f"{EXPECTED_DEVELOPMENT_INSTANCES}",
    flush=True,
)

print(
    f"Duplicate rows  : "
    f"{duplicate_label_rows}",
    flush=True,
)

print(
    f"Polygon rows    : "
    f"{segment_label_rows}",
    flush=True,
)

print(
    f"Data errors     : "
    f"{len(all_errors)}",
    flush=True,
)

if all_errors:
    with error_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label_path",
                "line_number",
                "line",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(all_errors)

    raise RuntimeError(
        "Invalid images or annotations were found. "
        "No empirical percentages should be used "
        "until these records are corrected.\n"
        f"Review: {error_csv}"
    )

if instance_id <= 0:
    raise RuntimeError(
        "No development annotations were found."
    )

if (
    instance_id
    != EXPECTED_DEVELOPMENT_INSTANCES
):
    print(
        "\nWARNING: The actual development-instance "
        "count does not equal 21,834.",
        flush=True,
    )
    print(
        "The generated table uses the actual "
        f"denominator N={instance_id}.",
        flush=True,
    )


# ============================================================
# Generate empirical Table 2
# ============================================================

stage("[5/6] Generating empirical Table 2")

summary_csv = (
    REPORTS
    / "table2_empirical_support_statistics.csv"
)

summary_json = (
    REPORTS
    / "table2_empirical_support_statistics.json"
)

summary_markdown = (
    REPORTS
    / "table2_empirical_support_statistics.md"
)

summary_latex = (
    REPORTS
    / "table2_empirical_support_statistics.tex"
)

key_findings_path = (
    REPORTS
    / "table2_key_findings_latex.tex"
)

metrics = [
    "q_min_lt_1",
    "q_min_lt_2",
    "q_area_lt_1",
    "q_area_lt_4",
    "q_max_lt_1_and_q_min_lt_1",
    "q_max_ge_1_and_q_min_lt_1",
    "q_max_lt_2_and_q_min_lt_2",
    "q_max_ge_2_and_q_min_lt_2",
]

summary_rows = []

for level, stride in LEVELS:
    total = summary[level]["total"]

    row: Dict[str, Any] = {
        "level": level,
        "stride": stride,
        "total_instances": total,
    }

    for metric in metrics:
        count = int(
            summary[level][metric]
        )

        row[f"{metric}_count"] = count
        row[f"{metric}_percent"] = (
            percentage(count, total)
        )

    summary_rows.append(row)

summary_fields = [
    "level",
    "stride",
    "total_instances",
]

for metric in metrics:
    summary_fields.extend(
        [
            f"{metric}_count",
            f"{metric}_percent",
        ]
    )

with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_fields,
    )

    writer.writeheader()
    writer.writerows(summary_rows)

summary_json.write_text(
    json.dumps(
        {
            "target_size": TARGET_SIZE,
            "development_instances": (
                instance_id
            ),
            "rows": summary_rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


markdown_lines = [
    (
        "| Level | Stride | "
        "q_min<1 | q_min<2 | "
        "q_area<1 | q_area<4 | "
        "q_max<1 & q_min<1 | "
        "q_max>=1 & q_min<1 | "
        "q_max<2 & q_min<2 | "
        "q_max>=2 & q_min<2 |"
    ),
    (
        "|---|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|"
    ),
]

for level, stride in LEVELS:
    total = summary[level]["total"]

    markdown_lines.append(
        "| {level} | {stride} | "
        "{a:.2f}% | {b:.2f}% | "
        "{c:.2f}% | {d:.2f}% | "
        "{e:.2f}% | {f:.2f}% | "
        "{g:.2f}% | {h:.2f}% |".format(
            level=level,
            stride=stride,
            a=percentage(
                summary[level][
                    "q_min_lt_1"
                ],
                total,
            ),
            b=percentage(
                summary[level][
                    "q_min_lt_2"
                ],
                total,
            ),
            c=percentage(
                summary[level][
                    "q_area_lt_1"
                ],
                total,
            ),
            d=percentage(
                summary[level][
                    "q_area_lt_4"
                ],
                total,
            ),
            e=percentage(
                summary[level][
                    "q_max_lt_1_and_q_min_lt_1"
                ],
                total,
            ),
            f=percentage(
                summary[level][
                    "q_max_ge_1_and_q_min_lt_1"
                ],
                total,
            ),
            g=percentage(
                summary[level][
                    "q_max_lt_2_and_q_min_lt_2"
                ],
                total,
            ),
            h=percentage(
                summary[level][
                    "q_max_ge_2_and_q_min_lt_2"
                ],
                total,
            ),
        )
    )

summary_markdown.write_text(
    "\n".join(markdown_lines) + "\n",
    encoding="utf-8",
)


latex_lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\renewcommand{\arraystretch}{1.22}",
    "",
    (
        r"\caption{Empirical two-dimensional "
        r"spatial-support insufficiency of WTMD "
        r"development instances at candidate "
        r"prediction levels after $960\times960$ "
        r"letterbox preprocessing. Percentages are "
        rf"calculated over all {instance_id:,} "
        r"training and validation instances; the "
        r"fixed test subset is excluded.}"
    ),
    r"\label{tab:stride-support}",
    "",
    r"\begin{tabularx}{\textwidth}{",
    r">{\centering\arraybackslash}p{0.10\textwidth}",
    r">{\centering\arraybackslash}p{0.10\textwidth}",
    r"*{4}{>{\centering\arraybackslash}X}",
    r"}",
    r"\toprule",
    (
        r"\multicolumn{6}{l}{\textit{Panel A: "
        r"marginal support-insufficiency "
        r"proportions}}\\"
    ),
    r"\midrule",
    r"\textbf{Level} &",
    r"\textbf{Stride} &",
    r"\textbf{$q_l^{\min}<1$} &",
    r"\textbf{$q_l^{\min}<2$} &",
    r"\textbf{$q_l^{\mathrm{area}}<1$} &",
    r"\textbf{$q_l^{\mathrm{area}}<4$} \\",
    r"\midrule",
]

for level, stride in LEVELS:
    total = summary[level]["total"]

    latex_lines.append(
        f"{level} & {stride} & "
        f"{latex_percentage(summary[level]['q_min_lt_1'], total)} & "
        f"{latex_percentage(summary[level]['q_min_lt_2'], total)} & "
        f"{latex_percentage(summary[level]['q_area_lt_1'], total)} & "
        f"{latex_percentage(summary[level]['q_area_lt_4'], total)} \\\\"
    )

latex_lines.extend(
    [
        r"\midrule",
        (
            r"\multicolumn{6}{l}{\textit{Panel B: "
            r"joint dominant--narrow support "
            r"states}}\\"
        ),
        r"\midrule",
        r"\textbf{Level} &",
        r"\textbf{Stride} &",
        (
            r"\textbf{\shortstack{$q_l^{\max}<1,$\\"
            r"$q_l^{\min}<1$}} &"
        ),
        (
            r"\textbf{\shortstack{$q_l^{\max}\geq1,$\\"
            r"$q_l^{\min}<1$}} &"
        ),
        (
            r"\textbf{\shortstack{$q_l^{\max}<2,$\\"
            r"$q_l^{\min}<2$}} &"
        ),
        (
            r"\textbf{\shortstack{$q_l^{\max}\geq2,$\\"
            r"$q_l^{\min}<2$}} \\"
        ),
        r"\midrule",
    ]
)

for level, stride in LEVELS:
    total = summary[level]["total"]

    latex_lines.append(
        f"{level} & {stride} & "
        f"{latex_percentage(summary[level]['q_max_lt_1_and_q_min_lt_1'], total)} & "
        f"{latex_percentage(summary[level]['q_max_ge_1_and_q_min_lt_1'], total)} & "
        f"{latex_percentage(summary[level]['q_max_lt_2_and_q_min_lt_2'], total)} & "
        f"{latex_percentage(summary[level]['q_max_ge_2_and_q_min_lt_2'], total)} \\\\"
    )

latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
)

summary_latex.write_text(
    "\n".join(latex_lines) + "\n",
    encoding="utf-8",
)


p2_total = summary["P2"]["total"]
p5_total = summary["P5"]["total"]

p5_qmin1 = percentage(
    summary["P5"]["q_min_lt_1"],
    p5_total,
)

p5_qarea4 = percentage(
    summary["P5"]["q_area_lt_4"],
    p5_total,
)

p5_discordant = percentage(
    summary["P5"][
        "q_max_ge_1_and_q_min_lt_1"
    ],
    p5_total,
)

p2_qmin1 = percentage(
    summary["P2"]["q_min_lt_1"],
    p2_total,
)

p2_qarea4 = percentage(
    summary["P2"]["q_area_lt_4"],
    p2_total,
)

p2_discordant = percentage(
    summary["P2"][
        "q_max_ge_1_and_q_min_lt_1"
    ],
    p2_total,
)

key_findings = (
    f"At P5, {p5_qmin1:.2f}\\% of the "
    f"development instances satisfy "
    f"$q_{{P5}}^{{\\min}}<1$, and "
    f"{p5_qarea4:.2f}\\% satisfy "
    f"$q_{{P5}}^{{\\mathrm{{area}}}}<4$. "
    f"Moreover, {p5_discordant:.2f}\\% exhibit "
    f"$q_{{P5}}^{{\\max}}\\geq1$ but "
    f"$q_{{P5}}^{{\\min}}<1$, showing that "
    f"dominant-direction support alone would "
    f"overstate the spatial representation "
    f"available to a subset of elongated targets. "
    f"The corresponding proportions decrease to "
    f"{p2_qmin1:.2f}\\%, {p2_qarea4:.2f}\\%, "
    f"and {p2_discordant:.2f}\\% at P2, "
    f"respectively.\n"
)

key_findings_path.write_text(
    key_findings,
    encoding="utf-8",
)


# ============================================================
# Final console result
# ============================================================

stage("[6/6] Completed empirical support statistics")

print(
    f"Development instances used: {instance_id}",
    flush=True,
)

print(
    "\nPanel A — marginal support insufficiency",
    flush=True,
)

print(
    "Level  Stride  qmin<1    qmin<2    "
    "qarea<1   qarea<4",
    flush=True,
)

for level, stride in LEVELS:
    total = summary[level]["total"]

    print(
        f"{level:<5}  "
        f"{stride:<6}  "
        f"{percentage(summary[level]['q_min_lt_1'], total):>7.2f}%  "
        f"{percentage(summary[level]['q_min_lt_2'], total):>7.2f}%  "
        f"{percentage(summary[level]['q_area_lt_1'], total):>7.2f}%  "
        f"{percentage(summary[level]['q_area_lt_4'], total):>7.2f}%",
        flush=True,
    )

print(
    "\nPanel B — joint dominant--narrow states",
    flush=True,
)

print(
    "Level  Stride  "
    "max<1,min<1  max>=1,min<1  "
    "max<2,min<2  max>=2,min<2",
    flush=True,
)

for level, stride in LEVELS:
    total = summary[level]["total"]

    print(
        f"{level:<5}  "
        f"{stride:<6}  "
        f"{percentage(summary[level]['q_max_lt_1_and_q_min_lt_1'], total):>11.2f}%  "
        f"{percentage(summary[level]['q_max_ge_1_and_q_min_lt_1'], total):>12.2f}%  "
        f"{percentage(summary[level]['q_max_lt_2_and_q_min_lt_2'], total):>11.2f}%  "
        f"{percentage(summary[level]['q_max_ge_2_and_q_min_lt_2'], total):>12.2f}%",
        flush=True,
    )

print("\nOutput files", flush=True)
print(f"LaTeX Table 2 : {summary_latex}", flush=True)
print(f"Summary CSV   : {summary_csv}", flush=True)
print(f"Summary JSON  : {summary_json}", flush=True)
print(f"Markdown      : {summary_markdown}", flush=True)
print(f"Key paragraph : {key_findings_path}", flush=True)
print(f"Raw geometry  : {geometry_csv}", flush=True)
print(f"Raw support   : {support_csv}", flush=True)
print(f"Quality report: {quality_path}", flush=True)

if (
    instance_id
    == EXPECTED_DEVELOPMENT_INSTANCES
):
    print(
        "\nInstance-count verification: PASS "
        "(21,834/21,834).",
        flush=True,
    )
else:
    print(
        "\nInstance-count verification: WARNING "
        f"({instance_id}/"
        f"{EXPECTED_DEVELOPMENT_INSTANCES}).",
        flush=True,
    )

print(
    "\nNo test images or test annotations were used.",
    flush=True,
)
