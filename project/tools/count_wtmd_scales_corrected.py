from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

# 无图形界面环境下也能保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_path(base_directory: Path, value: str | Path) -> Path:
    """
    将YAML中的绝对路径或相对路径转换为绝对路径。
    """
    path = Path(str(value))

    if path.is_absolute():
        return path.resolve()

    return (base_directory / path).resolve()


def load_dataset_yaml(yaml_path: Path) -> tuple[dict[str, Any], Path]:
    """
    读取Ultralytics数据集YAML，并确定数据集根目录。
    """
    with yaml_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("数据集YAML内容不是有效字典。")

    yaml_directory = yaml_path.parent.resolve()

    if config.get("path"):
        dataset_root = resolve_path(
            yaml_directory,
            config["path"],
        )
    else:
        dataset_root = yaml_directory

    return config, dataset_root


def collect_images(
    source: Any,
    dataset_root: Path,
) -> list[Path]:
    """
    收集YAML中指定的数据图像。

    支持：
    1. 图像文件夹；
    2. 图像路径列表txt；
    3. 单张图像；
    4. 多个路径构成的列表。
    """
    images: list[Path] = []

    if isinstance(source, (list, tuple)):
        for item in source:
            images.extend(
                collect_images(
                    source=item,
                    dataset_root=dataset_root,
                )
            )
        return images

    source_path = resolve_path(
        dataset_root,
        str(source),
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"YAML中的数据路径不存在：{source_path}"
        )

    if source_path.is_dir():
        for path in source_path.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                images.append(path.resolve())

        return images

    if source_path.suffix.lower() == ".txt":
        lines = source_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            candidate = Path(line)

            if not candidate.is_absolute():
                root_candidate = (
                    dataset_root / candidate
                ).resolve()

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
                    f"[警告] 图像列表路径不存在：{candidate}"
                )

        return images

    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [source_path.resolve()]

    raise ValueError(
        f"无法识别的数据路径类型：{source_path}"
    )


def image_to_label_path(image_path: Path) -> Path:
    """
    将标准YOLO图像路径转换为标签路径。

    例如：
        dataset/images/train/example.jpg

    转换为：
        dataset/labels/train/example.txt
    """
    parts = list(image_path.parts)

    image_indices = [
        index
        for index, part in enumerate(parts)
        if part.lower() == "images"
    ]

    if not image_indices:
        raise ValueError(
            f"图像路径中没有找到images目录：{image_path}"
        )

    # 使用路径中最后一个images目录
    image_index = image_indices[-1]
    parts[image_index] = "labels"

    return Path(*parts).with_suffix(".txt")


def classify_scale(scale_s: float) -> str:
    """
    按边界框最长边划分尺度。
    """
    if scale_s < 32:
        return "tiny"

    if scale_s < 96:
        return "small"

    return "medium"


def parse_yolo_label(
    label_path: Path,
    image_path: Path,
    image_width: int,
    image_height: int,
    target_size: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    读取YOLO标签，并按960×960 letterbox逻辑计算目标尺度。

    原图尺寸：
        W × H

    等比例缩放系数：
        r = min(target_size / W, target_size / H)

    缩放后的图像尺寸：
        round(W × r), round(H × r)

    padding只改变边界框位置，不改变边界框宽度和高度。
    """
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    if not label_path.exists():
        return valid_records, invalid_records

    label_text = label_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not label_text:
        return valid_records, invalid_records

    resize_ratio = min(
        target_size / image_width,
        target_size / image_height,
    )

    # 模拟Ultralytics LetterBox中的整数缩放尺寸
    resized_image_width = round(
        image_width * resize_ratio
    )

    resized_image_height = round(
        image_height * resize_ratio
    )

    scale_x = (
        resized_image_width / image_width
    )

    scale_y = (
        resized_image_height / image_height
    )

    for line_number, line in enumerate(
        label_text.splitlines(),
        start=1,
    ):
        parts = line.strip().split()

        base_invalid_record = {
            "split": split,
            "image": str(image_path),
            "label": str(label_path),
            "line_number": line_number,
            "raw_label": line.strip(),
        }

        if len(parts) < 5:
            invalid_records.append({
                **base_invalid_record,
                "reason": "标签列数少于5",
            })
            continue

        try:
            class_id = int(float(parts[0]))
            normalized_center_x = float(parts[1])
            normalized_center_y = float(parts[2])
            normalized_width = float(parts[3])
            normalized_height = float(parts[4])
        except ValueError:
            invalid_records.append({
                **base_invalid_record,
                "reason": "标签包含非数字内容",
            })
            continue

        values = [
            normalized_center_x,
            normalized_center_y,
            normalized_width,
            normalized_height,
        ]

        if not all(np.isfinite(value) for value in values):
            invalid_records.append({
                **base_invalid_record,
                "reason": "标签包含NaN或无穷值",
            })
            continue

        if normalized_width <= 0 or normalized_height <= 0:
            invalid_records.append({
                **base_invalid_record,
                "reason": "边界框宽度或高度小于等于0",
            })
            continue

        if normalized_width > 1 or normalized_height > 1:
            invalid_records.append({
                **base_invalid_record,
                "reason": "归一化宽度或高度大于1",
            })
            continue

        if not (
            0 <= normalized_center_x <= 1
            and 0 <= normalized_center_y <= 1
        ):
            invalid_records.append({
                **base_invalid_record,
                "reason": "边界框中心坐标不在0到1范围内",
            })
            continue

        x_min = (
            normalized_center_x
            - normalized_width / 2
        )

        x_max = (
            normalized_center_x
            + normalized_width / 2
        )

        y_min = (
            normalized_center_y
            - normalized_height / 2
        )

        y_max = (
            normalized_center_y
            + normalized_height / 2
        )

        tolerance = 1e-6

        if (
            x_min < -tolerance
            or y_min < -tolerance
            or x_max > 1 + tolerance
            or y_max > 1 + tolerance
        ):
            invalid_records.append({
                **base_invalid_record,
                "reason": "边界框超出图像归一化边界",
            })
            continue

        # 原图中的边界框宽度和高度
        original_box_width = (
            normalized_width * image_width
        )

        original_box_height = (
            normalized_height * image_height
        )

        # letterbox等比例缩放后的边界框宽度和高度
        resized_box_width = (
            original_box_width * scale_x
        )

        resized_box_height = (
            original_box_height * scale_y
        )

        scale_s = max(
            resized_box_width,
            resized_box_height,
        )

        if scale_s > target_size + tolerance:
            invalid_records.append({
                **base_invalid_record,
                "reason": (
                    f"缩放后最长边scale_s大于输入尺寸"
                    f"{target_size}"
                ),
            })
            continue

        scale_group = classify_scale(scale_s)

        valid_records.append({
            "split": split,
            "image": str(image_path),
            "label": str(label_path),
            "line_number": line_number,
            "class_id": class_id,
            "original_image_width": image_width,
            "original_image_height": image_height,
            "resize_ratio": resize_ratio,
            "resized_image_width": resized_image_width,
            "resized_image_height": resized_image_height,
            "normalized_box_width": normalized_width,
            "normalized_box_height": normalized_height,
            "original_box_width": original_box_width,
            "original_box_height": original_box_height,
            "box_width_960": resized_box_width,
            "box_height_960": resized_box_height,
            "scale_s": scale_s,
            "scale_group": scale_group,
        })

    return valid_records, invalid_records


def save_csv(
    file_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """
    保存CSV文件。
    """
    with file_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        if rows:
            writer.writerows(rows)


def create_scale_figure(
    scales: np.ndarray,
    output_directory: Path,
    target_size: int,
) -> tuple[Path, Path]:
    """
    生成完整0–960横轴直方图，并添加0–200局部放大图。
    """
    bin_width = 5

    full_bins = np.arange(
        0,
        target_size + bin_width,
        bin_width,
    )

    figure, axis = plt.subplots(
        figsize=(12, 6.2)
    )

    axis.hist(
        scales,
        bins=full_bins,
        rwidth=0.96,
    )

    axis.axvline(
        32,
        linestyle="--",
        linewidth=1.4,
    )

    axis.axvline(
        96,
        linestyle="--",
        linewidth=1.4,
    )

    # 完整输入范围，不再乘1.05扩展到1050
    axis.set_xlim(
        0,
        target_size,
    )

    major_ticks = list(
        np.arange(
            0,
            901,
            100,
        )
    )

    if target_size not in major_ticks:
        major_ticks.append(target_size)

    axis.set_xticks(major_ticks)

    axis.set_xlabel(
        r"Object long side $s$ "
        r"(pixels, after letterbox resizing to "
        r"$960\times960$)",
        fontsize=12,
    )

    axis.set_ylabel(
        "Instance count",
        fontsize=12,
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    # 使用坐标轴比例定位文字，防止与柱子严重重叠
    axis.text(
        32,
        0.97,
        r"$s=32$",
        transform=axis.get_xaxis_transform(),
        horizontalalignment="right",
        verticalalignment="top",
        rotation=90,
        fontsize=10,
    )

    axis.text(
        96,
        0.97,
        r"$s=96$",
        transform=axis.get_xaxis_transform(),
        horizontalalignment="right",
        verticalalignment="top",
        rotation=90,
        fontsize=10,
    )

    axis.text(
        16,
        0.84,
        "Tiny",
        transform=axis.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )

    axis.text(
        64,
        0.84,
        "Small",
        transform=axis.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )

    axis.text(
        170,
        0.84,
        "Medium",
        transform=axis.get_xaxis_transform(),
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )

    # --------------------------------------------------------
    # 0–200 px局部放大图
    # --------------------------------------------------------

    inset_axis = inset_axes(
        axis,
        width="52%",
        height="52%",
        loc="upper right",
        borderpad=1.3,
    )

    zoom_bins = np.arange(
        0,
        200 + bin_width,
        bin_width,
    )

    inset_axis.hist(
        scales,
        bins=zoom_bins,
        rwidth=0.96,
    )

    inset_axis.axvline(
        32,
        linestyle="--",
        linewidth=1.0,
    )

    inset_axis.axvline(
        96,
        linestyle="--",
        linewidth=1.0,
    )

    inset_axis.set_xlim(
        0,
        200,
    )

    inset_axis.set_xticks(
        [0, 32, 50, 96, 150, 200]
    )

    inset_axis.set_title(
        "Enlarged view: 0–200 px",
        fontsize=9,
    )

    inset_axis.tick_params(
        axis="both",
        labelsize=8,
    )

    inset_axis.text(
        32,
        0.96,
        "32",
        transform=inset_axis.get_xaxis_transform(),
        horizontalalignment="right",
        verticalalignment="top",
        rotation=90,
        fontsize=8,
    )

    inset_axis.text(
        96,
        0.96,
        "96",
        transform=inset_axis.get_xaxis_transform(),
        horizontalalignment="right",
        verticalalignment="top",
        rotation=90,
        fontsize=8,
    )

    # 在主图中标记局部放大范围0–200
    main_y_min, main_y_max = axis.get_ylim()

    zoom_rectangle = Rectangle(
        (0, main_y_min),
        200,
        main_y_max - main_y_min,
        fill=False,
        linestyle=":",
        linewidth=0.8,
    )

    axis.add_patch(zoom_rectangle)

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.16,
        top=0.97,
    )

    pdf_path = (
        output_directory
        / "Figure2_scale_distribution_corrected.pdf"
    )

    png_path = (
        output_directory
        / "Figure2_scale_distribution_corrected.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(figure)

    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "按照960×960 letterbox等比例缩放重新统计"
            "WTMD训练集和验证集目标尺度。"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Ultralytics数据集YAML路径",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="目标输入尺寸，默认960",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="参与统计的数据划分，默认train val",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="结果保存目录",
    )

    args = parser.parse_args()

    yaml_path = args.data.resolve()
    output_directory = args.out.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config, dataset_root = load_dataset_yaml(
        yaml_path
    )

    print("")
    print("=" * 75)
    print("WTMD目标尺度重新统计")
    print("=" * 75)
    print(f"数据集YAML：{yaml_path}")
    print(f"数据集根目录：{dataset_root}")
    print(f"输入尺寸：{args.imgsz} × {args.imgsz}")
    print(f"统计划分：{', '.join(args.splits)}")
    print("=" * 75)

    all_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    split_image_counts: Counter[str] = Counter()
    split_background_counts: Counter[str] = Counter()
    split_instance_counts: Counter[str] = Counter()
    split_missing_label_counts: Counter[str] = Counter()
    unreadable_images: list[dict[str, Any]] = []

    for split in args.splits:
        if split not in config:
            raise KeyError(
                f"数据集YAML中不存在划分字段：{split}"
            )

        image_paths = sorted(
            set(
                collect_images(
                    source=config[split],
                    dataset_root=dataset_root,
                )
            )
        )

        split_image_counts[split] = len(image_paths)

        print("")
        print(
            f"[{split}] 找到图像："
            f"{len(image_paths)} 张"
        )

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
            except Exception as error:
                unreadable_images.append({
                    "split": split,
                    "image": str(image_path),
                    "reason": str(error),
                })
                continue

            try:
                label_path = image_to_label_path(
                    image_path
                )
            except ValueError as error:
                invalid_records.append({
                    "split": split,
                    "image": str(image_path),
                    "label": "",
                    "line_number": "",
                    "raw_label": "",
                    "reason": str(error),
                })
                continue

            if not label_path.exists():
                split_missing_label_counts[split] += 1
                split_background_counts[split] += 1
                continue

            valid_rows, invalid_rows = parse_yolo_label(
                label_path=label_path,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
                target_size=args.imgsz,
                split=split,
            )

            invalid_records.extend(invalid_rows)

            if not valid_rows:
                split_background_counts[split] += 1
            else:
                split_instance_counts[split] += len(valid_rows)
                all_records.extend(valid_rows)

    # --------------------------------------------------------
    # 保存异常记录
    # --------------------------------------------------------

    invalid_csv = (
        output_directory
        / "invalid_labels.csv"
    )

    invalid_fields = [
        "split",
        "image",
        "label",
        "line_number",
        "raw_label",
        "reason",
    ]

    save_csv(
        file_path=invalid_csv,
        rows=invalid_records,
        fieldnames=invalid_fields,
    )

    unreadable_csv = (
        output_directory
        / "unreadable_images.csv"
    )

    save_csv(
        file_path=unreadable_csv,
        rows=unreadable_images,
        fieldnames=[
            "split",
            "image",
            "reason",
        ],
    )

    if not all_records:
        raise RuntimeError(
            "没有读取到有效目标框，请检查数据路径和标签。"
        )

    scales = np.asarray(
        [
            float(record["scale_s"])
            for record in all_records
        ],
        dtype=float,
    )

    scale_counter = Counter(
        record["scale_group"]
        for record in all_records
    )

    total_instances = len(all_records)

    tiny_count = scale_counter["tiny"]
    small_count = scale_counter["small"]
    medium_count = scale_counter["medium"]
    tiny_small_count = tiny_count + small_count

    tiny_ratio = (
        tiny_count / total_instances * 100
    )

    small_ratio = (
        small_count / total_instances * 100
    )

    medium_ratio = (
        medium_count / total_instances * 100
    )

    tiny_small_ratio = (
        tiny_small_count
        / total_instances
        * 100
    )

    scale_min = float(np.min(scales))
    scale_max = float(np.max(scales))
    scale_mean = float(np.mean(scales))
    scale_median = float(np.median(scales))

    greater_than_200 = int(
        np.sum(scales > 200)
    )

    greater_than_300 = int(
        np.sum(scales > 300)
    )

    greater_than_500 = int(
        np.sum(scales > 500)
    )

    greater_than_900 = int(
        np.sum(scales > 900)
    )

    greater_than_960 = int(
        np.sum(scales > args.imgsz)
    )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print("")
    print("=" * 75)
    print("数据集统计")
    print("=" * 75)

    for split in args.splits:
        print(
            f"{split:>10} 图像数："
            f"{split_image_counts[split]}"
        )

        print(
            f"{split:>10} 实例数："
            f"{split_instance_counts[split]}"
        )

        print(
            f"{split:>10} 背景图："
            f"{split_background_counts[split]}"
        )

        print(
            f"{split:>10} 缺少标签文件："
            f"{split_missing_label_counts[split]}"
        )

    print("-" * 75)
    print("Table 2目标尺度统计")
    print("-" * 75)
    print(
        f"{'Tiny':>10}："
        f"{tiny_count:>7}，"
        f"{tiny_ratio:>7.2f}%"
    )

    print(
        f"{'Small':>10}："
        f"{small_count:>7}，"
        f"{small_ratio:>7.2f}%"
    )

    print(
        f"{'Medium':>10}："
        f"{medium_count:>7}，"
        f"{medium_ratio:>7.2f}%"
    )

    print(
        f"{'Total':>10}："
        f"{total_instances:>7}，"
        f"100.00%"
    )

    print(
        f"{'Tiny+Small':>10}："
        f"{tiny_small_count:>7}，"
        f"{tiny_small_ratio:>7.2f}%"
    )

    print("-" * 75)
    print("尺度范围检查")
    print("-" * 75)
    print(f"最小尺度 s_min：{scale_min:.4f} px")
    print(f"最大尺度 s_max：{scale_max:.4f} px")
    print(f"平均尺度：{scale_mean:.4f} px")
    print(f"中位尺度：{scale_median:.4f} px")
    print(f"s > 200 px：{greater_than_200}")
    print(f"s > 300 px：{greater_than_300}")
    print(f"s > 500 px：{greater_than_500}")
    print(f"s > 900 px：{greater_than_900}")
    print(f"s > 960 px：{greater_than_960}")
    print(f"异常标签记录：{len(invalid_records)}")
    print(f"无法读取的图像：{len(unreadable_images)}")
    print("=" * 75)

    # --------------------------------------------------------
    # 保存逐目标详细CSV
    # --------------------------------------------------------

    detail_csv = (
        output_directory
        / "scale_details.csv"
    )

    detail_fields = [
        "split",
        "image",
        "label",
        "line_number",
        "class_id",
        "original_image_width",
        "original_image_height",
        "resize_ratio",
        "resized_image_width",
        "resized_image_height",
        "normalized_box_width",
        "normalized_box_height",
        "original_box_width",
        "original_box_height",
        "box_width_960",
        "box_height_960",
        "scale_s",
        "scale_group",
    ]

    save_csv(
        file_path=detail_csv,
        rows=all_records,
        fieldnames=detail_fields,
    )

    # --------------------------------------------------------
    # 保存Table 2汇总CSV
    # --------------------------------------------------------

    summary_csv = (
        output_directory
        / "scale_summary.csv"
    )

    summary_rows = [
        {
            "Scale group": "Tiny",
            "Long-side-based definition": "s < 32",
            "Count": tiny_count,
            "Ratio (%)": f"{tiny_ratio:.2f}",
        },
        {
            "Scale group": "Small",
            "Long-side-based definition": "32 <= s < 96",
            "Count": small_count,
            "Ratio (%)": f"{small_ratio:.2f}",
        },
        {
            "Scale group": "Medium",
            "Long-side-based definition": "s >= 96",
            "Count": medium_count,
            "Ratio (%)": f"{medium_ratio:.2f}",
        },
        {
            "Scale group": "Total",
            "Long-side-based definition": "-",
            "Count": total_instances,
            "Ratio (%)": "100.00",
        },
    ]

    save_csv(
        file_path=summary_csv,
        rows=summary_rows,
        fieldnames=[
            "Scale group",
            "Long-side-based definition",
            "Count",
            "Ratio (%)",
        ],
    )

    # --------------------------------------------------------
    # 保存文本报告
    # --------------------------------------------------------

    report_path = (
        output_directory
        / "scale_report.txt"
    )

    report_text = f"""WTMD target-scale statistics
========================================

Preprocessing:
Canonical letterbox input size: {args.imgsz} x {args.imgsz}
Resize ratio: r = min({args.imgsz} / W, {args.imgsz} / H)
Target scale: s = max(w', h')

Scale definition:
Tiny: s < 32
Small: 32 <= s < 96
Medium: s >= 96

Results:
Tiny: {tiny_count} ({tiny_ratio:.2f}%)
Small: {small_count} ({small_ratio:.2f}%)
Medium: {medium_count} ({medium_ratio:.2f}%)
Total: {total_instances} (100.00%)
Tiny + Small: {tiny_small_count} ({tiny_small_ratio:.2f}%)

Scale range:
Minimum s: {scale_min:.4f} px
Maximum s: {scale_max:.4f} px
Mean s: {scale_mean:.4f} px
Median s: {scale_median:.4f} px
s > 200 px: {greater_than_200}
s > 300 px: {greater_than_300}
s > 500 px: {greater_than_500}
s > 900 px: {greater_than_900}
s > {args.imgsz} px: {greater_than_960}

Data quality:
Invalid label rows: {len(invalid_records)}
Unreadable images: {len(unreadable_images)}
"""

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 生成修正后的Figure 2
    # --------------------------------------------------------

    pdf_path, png_path = create_scale_figure(
        scales=scales,
        output_directory=output_directory,
        target_size=args.imgsz,
    )

    print("")
    print("结果文件已生成：")
    print(f"1. Table 2汇总：{summary_csv}")
    print(f"2. 每个目标详情：{detail_csv}")
    print(f"3. 文本报告：{report_path}")
    print(f"4. 异常标签：{invalid_csv}")
    print(f"5. 无法读取图像：{unreadable_csv}")
    print(f"6. Figure 2 PDF：{pdf_path}")
    print(f"7. Figure 2 PNG：{png_path}")

    if invalid_records:
        print("")
        print(
            "[警告] 检测到异常标签，"
            "请打开invalid_labels.csv逐项检查。"
        )

    if greater_than_960 > 0:
        raise RuntimeError(
            f"检测到{greater_than_960}个目标的"
            f"scale_s大于{args.imgsz}，统计结果不可直接使用。"
        )


if __name__ == "__main__":
    main()
