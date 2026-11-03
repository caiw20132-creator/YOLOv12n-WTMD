from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def resolve_path(base: Path, value: str | Path) -> Path:
    """将相对路径转换为绝对路径。"""
    path = Path(str(value))

    if path.is_absolute():
        return path.resolve()

    return (base / path).resolve()


def collect_images(source, dataset_root: Path) -> list[Path]:
    """
    根据YAML中的train/val字段收集图像。
    支持：
    1. 图像目录
    2. 图像列表txt
    3. 单张图像
    4. 多个路径组成的列表
    """
    images: list[Path] = []

    if isinstance(source, (list, tuple)):
        for item in source:
            images.extend(collect_images(item, dataset_root))
        return images

    source_path = resolve_path(dataset_root, source)

    if not source_path.exists():
        raise FileNotFoundError(
            f"YAML中的数据路径不存在：{source_path}"
        )

    if source_path.is_dir():
        for path in source_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path.resolve())

    elif source_path.suffix.lower() == ".txt":
        lines = source_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            candidate = Path(line)

            if not candidate.is_absolute():
                # 优先按数据集根目录解析
                root_candidate = (dataset_root / candidate).resolve()

                # 其次按图像列表文件所在目录解析
                list_candidate = (
                    source_path.parent / candidate
                ).resolve()

                if root_candidate.exists():
                    candidate = root_candidate
                else:
                    candidate = list_candidate

            if candidate.exists():
                images.append(candidate.resolve())
            else:
                print(
                    f"[警告] 图像列表中的路径不存在：{candidate}"
                )

    elif source_path.suffix.lower() in IMAGE_EXTENSIONS:
        images.append(source_path.resolve())

    else:
        raise ValueError(
            f"无法识别的数据路径类型：{source_path}"
        )

    return images


def image_to_label_path(image_path: Path) -> Path:
    """
    将标准YOLO目录：
        .../images/train/xxx.jpg
    转换为：
        .../labels/train/xxx.txt
    """
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, part in enumerate(parts)
        if part.lower() == "images"
    ]

    if not image_indices:
        raise ValueError(
            f"图像路径中未找到images目录：{image_path}"
        )

    index = image_indices[-1]
    parts[index] = "labels"

    return Path(*parts).with_suffix(".txt")


def load_dataset_yaml(yaml_path: Path):
    """读取Ultralytics数据集YAML。"""
    with yaml_path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("数据集YAML内容无效。")

    yaml_directory = yaml_path.parent

    if config.get("path"):
        dataset_root = resolve_path(
            yaml_directory,
            config["path"]
        )
    else:
        dataset_root = yaml_directory.resolve()

    return config, dataset_root


def parse_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
    image_size: int
) -> Iterable[dict]:
    """
    读取YOLO标签并按照960×960 letterbox等比例缩放计算目标尺寸。

    Ultralytics LetterBox的核心缩放比例：
        r = min(imgsz / W, imgsz / H)

    padding只改变边界框位置，不改变边界框宽高。
    """
    if not label_path.exists():
        return []

    text = label_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        return []

    # 等比例缩放系数
    r = min(
        image_size / image_width,
        image_size / image_height
    )

    # 模拟实际图像缩放后的整数尺寸
    resized_width = round(image_width * r)
    resized_height = round(image_height * r)

    scale_x = resized_width / image_width
    scale_y = resized_height / image_height

    records = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):
        parts = line.strip().split()

        if len(parts) < 5:
            print(
                f"[警告] 标签格式不足5列："
                f"{label_path}，第{line_number}行"
            )
            continue

        try:
            class_id = int(float(parts[0]))
            normalized_width = float(parts[3])
            normalized_height = float(parts[4])
        except ValueError:
            print(
                f"[警告] 标签包含非数字内容："
                f"{label_path}，第{line_number}行"
            )
            continue

        if (
            normalized_width <= 0
            or normalized_height <= 0
        ):
            print(
                f"[警告] 边界框宽高无效："
                f"{label_path}，第{line_number}行"
            )
            continue

        # 原图中的框宽和框高
        original_box_width = (
            normalized_width * image_width
        )
        original_box_height = (
            normalized_height * image_height
        )

        # letterbox等比例缩放后的框宽和框高
        resized_box_width = (
            original_box_width * scale_x
        )
        resized_box_height = (
            original_box_height * scale_y
        )

        # 本文采用最长边尺度定义
        scale_s = max(
            resized_box_width,
            resized_box_height
        )

        if scale_s < 32:
            scale_group = "tiny"
        elif scale_s < 96:
            scale_group = "small"
        else:
            scale_group = "medium"

        records.append({
            "class_id": class_id,
            "box_width_960": resized_box_width,
            "box_height_960": resized_box_height,
            "scale_s": scale_s,
            "scale_group": scale_group
        })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "按照960×960 letterbox预处理重新统计"
            "WTMD目标尺度分布。"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Ultralytics数据集YAML路径"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="目标输入尺寸，默认960"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="参与统计的数据划分，默认train val"
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="统计结果保存目录"
    )

    args = parser.parse_args()

    yaml_path = args.data.resolve()
    output_directory = args.out.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    config, dataset_root = load_dataset_yaml(
        yaml_path
    )

    print("=" * 70)
    print(f"数据集YAML：{yaml_path}")
    print(f"数据集根目录：{dataset_root}")
    print(f"输入尺寸：{args.imgsz} × {args.imgsz}")
    print(f"统计划分：{', '.join(args.splits)}")
    print("=" * 70)

    all_records = []
    split_image_counts = {}
    split_background_counts = {}
    missing_label_count = 0

    for split in args.splits:
        if split not in config:
            raise KeyError(
                f"数据集YAML中不存在划分字段：{split}"
            )

        image_paths = sorted(
            set(
                collect_images(
                    config[split],
                    dataset_root
                )
            )
        )

        split_image_counts[split] = len(image_paths)
        split_background_counts[split] = 0

        print(
            f"\n[{split}] 找到图像："
            f"{len(image_paths)} 张"
        )

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except Exception as error:
                print(
                    f"[警告] 无法读取图像："
                    f"{image_path}，原因：{error}"
                )
                continue

            try:
                label_path = image_to_label_path(
                    image_path
                )
            except ValueError as error:
                print(f"[警告] {error}")
                continue

            if not label_path.exists():
                missing_label_count += 1
                split_background_counts[split] += 1
                continue

            records = list(
                parse_yolo_labels(
                    label_path=label_path,
                    image_width=image_width,
                    image_height=image_height,
                    image_size=args.imgsz
                )
            )

            if len(records) == 0:
                split_background_counts[split] += 1

            for record in records:
                record.update({
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path),
                    "original_width": image_width,
                    "original_height": image_height
                })
                all_records.append(record)

    if not all_records:
        raise RuntimeError(
            "没有读取到有效目标框，请检查数据路径和标签格式。"
        )

    scales = np.array(
        [
            record["scale_s"]
            for record in all_records
        ],
        dtype=float
    )

    counts = {
        "tiny": sum(
            record["scale_group"] == "tiny"
            for record in all_records
        ),
        "small": sum(
            record["scale_group"] == "small"
            for record in all_records
        ),
        "medium": sum(
            record["scale_group"] == "medium"
            for record in all_records
        )
    }

    total_instances = len(all_records)
    tiny_small_count = (
        counts["tiny"] + counts["small"]
    )

    scale_max = float(np.max(scales))
    scale_min = float(np.min(scales))
    scale_mean = float(np.mean(scales))
    scale_median = float(np.median(scales))

    greater_than_200 = int(
        np.sum(scales > 200)
    )
    greater_than_300 = int(
        np.sum(scales > 300)
    )

    # 横轴右侧预留5%空间，并向上取整至50的整数倍
    axis_max = (
        math.ceil(
            (scale_max * 1.05) / 50
        ) * 50
    )

    axis_max = max(axis_max, 100)

    print("\n")
    print("=" * 70)
    print("WTMD目标尺度统计结果")
    print("=" * 70)

    for split in args.splits:
        print(
            f"{split:>10} 图像数："
            f"{split_image_counts[split]}"
        )
        print(
            f"{split:>10} 背景图："
            f"{split_background_counts[split]}"
        )

    print("-" * 70)

    for group in ["tiny", "small", "medium"]:
        ratio = (
            counts[group]
            / total_instances
            * 100
        )

        print(
            f"{group:>10}："
            f"{counts[group]:>6}，"
            f"{ratio:>7.2f}%"
        )

    print(
        f"{'total':>10}："
        f"{total_instances:>6}，100.00%"
    )

    print(
        f"{'tiny+small':>10}："
        f"{tiny_small_count:>6}，"
        f"{tiny_small_count / total_instances * 100:>7.2f}%"
    )

    print("-" * 70)
    print(f"最小尺度 s_min：{scale_min:.4f} px")
    print(f"最大尺度 s_max：{scale_max:.4f} px")
    print(f"平均尺度：{scale_mean:.4f} px")
    print(f"中位尺度：{scale_median:.4f} px")
    print(f"s > 200 px：{greater_than_200} 个")
    print(f"s > 300 px：{greater_than_300} 个")
    print(f"Figure 2建议横轴上限：{axis_max} px")
    print(f"缺少标签文件的图像数：{missing_label_count}")
    print("=" * 70)

    # --------------------------------------------------------
    # 保存逐目标详细数据
    # --------------------------------------------------------
    detail_csv = (
        output_directory
        / "scale_details.csv"
    )

    fieldnames = [
        "split",
        "image",
        "label",
        "class_id",
        "original_width",
        "original_height",
        "box_width_960",
        "box_height_960",
        "scale_s",
        "scale_group"
    ]

    with detail_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    # --------------------------------------------------------
    # 保存汇总表
    # --------------------------------------------------------
    summary_csv = (
        output_directory
        / "scale_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Scale group",
            "Definition",
            "Count",
            "Ratio (%)"
        ])

        writer.writerow([
            "Tiny",
            "s < 32",
            counts["tiny"],
            f"{counts['tiny'] / total_instances * 100:.2f}"
        ])

        writer.writerow([
            "Small",
            "32 <= s < 96",
            counts["small"],
            f"{counts['small'] / total_instances * 100:.2f}"
        ])

        writer.writerow([
            "Medium",
            "s >= 96",
            counts["medium"],
            f"{counts['medium'] / total_instances * 100:.2f}"
        ])

        writer.writerow([
            "Total",
            "-",
            total_instances,
            "100.00"
        ])

    # --------------------------------------------------------
    # 绘制Figure 2
    # --------------------------------------------------------
    bin_width = 5

    bins = np.arange(
        0,
        axis_max + bin_width,
        bin_width
    )

    plt.figure(figsize=(10, 5.5))

    plt.hist(
        scales,
        bins=bins
    )

    plt.axvline(
        32,
        linestyle="--",
        linewidth=1.2
    )

    plt.axvline(
        96,
        linestyle="--",
        linewidth=1.2
    )

    plt.text(
        32,
        plt.ylim()[1] * 0.95,
        "s = 32",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right"
    )

    plt.text(
        96,
        plt.ylim()[1] * 0.95,
        "s = 96",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right"
    )

    plt.xlim(0, axis_max)

    plt.xticks(
        np.arange(
            0,
            axis_max + 1,
            50
        )
    )

    plt.xlabel(
        "Object long side s "
        "(pixels, after letterbox resizing to 960 × 960)"
    )

    plt.ylabel("Instance count")

    plt.tight_layout()

    figure_pdf = (
        output_directory
        / "Figure2_scale_distribution.pdf"
    )

    figure_png = (
        output_directory
        / "Figure2_scale_distribution.png"
    )

    plt.savefig(
        figure_pdf,
        bbox_inches="tight"
    )

    plt.savefig(
        figure_png,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print("\n结果文件已保存：")
    print(f"1. 详细数据：{detail_csv}")
    print(f"2. 汇总数据：{summary_csv}")
    print(f"3. PDF图：{figure_pdf}")
    print(f"4. PNG图：{figure_png}")


if __name__ == "__main__":
    main()
