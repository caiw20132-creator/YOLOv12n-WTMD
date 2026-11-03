from pathlib import Path
import csv
import pickle
import yaml
import cv2
import numpy as np
import torch

from ultralytics import YOLO


# ============================================================
# A. 固定路径
# ============================================================

ROOT = Path(r"D:\YOLO-WTB")

DATA = (
    ROOT
    / "data"
    / "wtb_damage_4021_1151.yaml"
)

# 当前 controlled hierarchy 实验使用的 baseline
BASELINE_PT = (
    ROOT
    / "runs_m4_schemeA_official_8x3"
    / "train"
    / "Baseline_YOLOv12n_seed0"
    / "weights"
    / "best.pt"
)

# 刚刚训练完成的 H11
H11_PT = (
    ROOT
    / "runs_controlled_hierarchy_minimal"
    / "train"
    / "H11_Reallocate_P234_seed0"
    / "weights"
    / "best.pt"
)

OUT = (
    ROOT
    / "runs_controlled_hierarchy_minimal"
    / "verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PRED_CACHE = OUT / "pred_cache"

PRED_CACHE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# B. 固定论文评价协议
# ============================================================

IMGSZ = 960
DEVICE = 0

CONF = 0.001
NMS_IOU = 0.50
MAX_DET = 10000

IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05
)

GROUP_NAMES = [
    "tiny",
    "small",
    "medium-to-large"
]

# 论文 Table 10 当前固定 test 数量
EXPECTED_COUNTS = {
    "tiny": 825,
    "small": 967,
    "medium-to-large": 509,
}

EXPECTED_TOTAL = 2301


# ============================================================
# C. 文件检查
# ============================================================

for p in [
    DATA,
    BASELINE_PT,
    H11_PT,
]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing file: {p}"
        )

print("=" * 100)
print("PATH CHECK PASSED")
print("=" * 100)

print("DATA       =", DATA)
print("BASELINE   =", BASELINE_PT)
print("H11        =", H11_PT)
print("OUTPUT     =", OUT)


# ============================================================
# D. Part 1:
#    fixed-test overall metrics
# ============================================================

OVERALL_CSV = (
    OUT
    / "fixed_test_overall_baseline_vs_h11.csv"
)


def run_standard_test(
    name,
    weight,
    hierarchy,
):
    print("\n" + "=" * 100)
    print(
        f"FIXED-TEST STANDARD EVALUATION: {name}"
    )
    print("=" * 100)

    model = YOLO(str(weight))

    metrics = model.val(
        data=str(DATA),
        split="test",
        imgsz=IMGSZ,
        batch=1,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        augment=False,
        rect=False,
        half=False,
        device=DEVICE,
        workers=0,
        plots=False,
        save_json=False,
        project=str(OUT / "standard_test"),
        name=name,
        exist_ok=True,
    )

    params_m = (
        sum(
            p.numel()
            for p in model.model.parameters()
        )
        / 1e6
    )

    stride = [
        int(round(float(x)))
        for x in model.model.model[-1].stride
    ]

    row = {
        "Model": name,
        "Hierarchy": hierarchy,
        "K": len(stride),
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "AP50": float(metrics.box.map50),
        "AP50_95": float(metrics.box.map),
        "Params_M": float(params_m),
        "Strides": "/".join(
            str(x) for x in stride
        ),
    }

    print(
        f"{name}: "
        f"P={row['Precision']:.4f}, "
        f"R={row['Recall']:.4f}, "
        f"AP50={row['AP50']:.4f}, "
        f"AP50:95={row['AP50_95']:.4f}, "
        f"Params={row['Params_M']:.4f}M, "
        f"Stride={row['Strides']}"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


overall_rows = []

overall_rows.append(
    run_standard_test(
        "YOLOv12n_Baseline",
        BASELINE_PT,
        "P3/P4/P5",
    )
)

overall_rows.append(
    run_standard_test(
        "H11_Reallocation",
        H11_PT,
        "P2/P3/P4",
    )
)

with OVERALL_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            overall_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(overall_rows)


# ============================================================
# E. 解析 dataset.yaml 的 test split
# ============================================================

with DATA.open(
    "r",
    encoding="utf-8"
) as f:

    data_cfg = yaml.safe_load(f)


def resolve_root():
    root_value = data_cfg.get(
        "path",
        ""
    )

    if not root_value:
        return DATA.parent

    p = Path(str(root_value))

    if p.is_absolute() and p.exists():
        return p

    candidates = [
        DATA.parent / p,
        ROOT / p,
        p,
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return (
        DATA.parent / p
    ).resolve()


DATASET_ROOT = resolve_root()


def resolve_path(
    item,
    reference=None,
):
    p = Path(str(item))

    if p.is_absolute():
        return p

    candidates = []

    if reference is not None:
        candidates.append(
            reference / p
        )

    candidates.extend([
        DATASET_ROOT / p,
        DATA.parent / p,
        ROOT / p,
    ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    return candidates[0].resolve()


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def images_from_split_entry(
    entry,
):
    entry_path = resolve_path(entry)

    if not entry_path.exists():
        raise FileNotFoundError(
            f"test split path not found: "
            f"{entry_path}"
        )

    images = []

    if (
        entry_path.is_file()
        and entry_path.suffix.lower() == ".txt"
    ):

        lines = entry_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            p = Path(line)

            if not p.is_absolute():
                p = resolve_path(
                    line,
                    entry_path.parent
                )

            if p.suffix.lower() in IMAGE_EXTS:
                images.append(
                    p.resolve()
                )

    elif entry_path.is_dir():

        images = [
            p.resolve()
            for p in entry_path.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTS
            )
        ]

    elif (
        entry_path.is_file()
        and entry_path.suffix.lower()
        in IMAGE_EXTS
    ):

        images = [
            entry_path.resolve()
        ]

    else:
        raise RuntimeError(
            f"Unsupported test entry: "
            f"{entry_path}"
        )

    return images


test_spec = data_cfg.get(
    "test",
    None
)

if test_spec is None:
    raise RuntimeError(
        "DATA YAML 中没有 test 字段。"
    )

if isinstance(
    test_spec,
    (list, tuple)
):
    test_images = []

    for x in test_spec:
        test_images.extend(
            images_from_split_entry(x)
        )
else:
    test_images = images_from_split_entry(
        test_spec
    )

# 去重
test_images = sorted(
    set(test_images)
)

if len(test_images) == 0:
    raise RuntimeError(
        "没有找到 test images。"
    )

print("\n" + "=" * 100)
print("FIXED TEST DATASET")
print("=" * 100)

print(
    "Dataset root =",
    DATASET_ROOT
)

print(
    "Test images  =",
    len(test_images)
)


# ============================================================
# F. image -> label 映射
# ============================================================

def image_to_label(
    image_path
):
    parts = list(
        image_path.parts
    )

    image_indices = [
        i
        for i, x
        in enumerate(parts)
        if x.lower() == "images"
    ]

    if image_indices:

        idx = image_indices[-1]

        parts[idx] = "labels"

        label_path = Path(
            *parts
        ).with_suffix(".txt")

        return label_path

    # fallback:
    candidate = (
        image_path.parent.parent
        / "labels"
        / (
            image_path.stem
            + ".txt"
        )
    )

    return candidate


# ============================================================
# G. IoU
# ============================================================

def iou_one_to_many(
    box,
    boxes
):
    if len(boxes) == 0:
        return np.zeros(
            (0,),
            dtype=np.float32
        )

    x1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    inter_w = np.maximum(
        0.0,
        x2 - x1
    )

    inter_h = np.maximum(
        0.0,
        y2 - y1
    )

    inter = (
        inter_w
        * inter_h
    )

    area1 = max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )

    area2 = (
        np.maximum(
            0.0,
            boxes[:, 2]
            - boxes[:, 0]
        )
        *
        np.maximum(
            0.0,
            boxes[:, 3]
            - boxes[:, 1]
        )
    )

    union = (
        area1
        + area2
        - inter
    )

    return (
        inter
        / np.maximum(
            union,
            1e-12
        )
    )


# ============================================================
# H. 读取 GT，严格按照论文 long-side grouping
#
# after 960 letterbox:
#
# tiny:
#       s_long < 32
#
# small:
#       32 <= s_long < 96
#
# medium-to-large:
#       s_long >= 96
# ============================================================

gt_db = {}

group_counts = {
    0: 0,
    1: 0,
    2: 0,
}

background_images = 0


for n, img_path in enumerate(
    test_images,
    1
):

    im = cv2.imread(
        str(img_path)
    )

    if im is None:
        raise RuntimeError(
            f"无法读取图片: "
            f"{img_path}"
        )

    h, w = im.shape[:2]

    gain = min(
        IMGSZ / w,
        IMGSZ / h
    )

    label_path = image_to_label(
        img_path
    )

    boxes = []
    groups = []

    if (
        label_path.exists()
        and label_path.stat().st_size > 0
    ):

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            cols = line.strip().split()

            if len(cols) < 5:
                continue

            cls_id = int(
                float(cols[0])
            )

            xc = float(cols[1])
            yc = float(cols[2])
            bw = float(cols[3])
            bh = float(cols[4])

            # 原图像素尺度
            bw_px = bw * w
            bh_px = bh * h

            # 960 letterbox 后的目标尺寸
            bw_960 = (
                bw_px
                * gain
            )

            bh_960 = (
                bh_px
                * gain
            )

            s_long = max(
                bw_960,
                bh_960
            )

            if s_long < 32.0:
                group = 0

            elif s_long < 96.0:
                group = 1

            else:
                group = 2

            group_counts[group] += 1

            # GT xyxy in ORIGINAL IMAGE coordinates
            x1 = (
                xc - bw / 2
            ) * w

            y1 = (
                yc - bh / 2
            ) * h

            x2 = (
                xc + bw / 2
            ) * w

            y2 = (
                yc + bh / 2
            ) * h

            boxes.append([
                x1,
                y1,
                x2,
                y2,
            ])

            groups.append(
                group
            )

    if len(boxes) == 0:

        background_images += 1

        box_array = np.zeros(
            (0, 4),
            dtype=np.float32
        )

        group_array = np.zeros(
            (0,),
            dtype=np.int64
        )

    else:

        box_array = np.asarray(
            boxes,
            dtype=np.float32
        )

        group_array = np.asarray(
            groups,
            dtype=np.int64
        )

    gt_db[
        str(img_path.resolve())
    ] = {
        "boxes": box_array,
        "groups": group_array,
    }


print("\n" + "=" * 100)
print("SCALE GROUP CHECK")
print("=" * 100)

actual_counts = {
    "tiny":
        group_counts[0],

    "small":
        group_counts[1],

    "medium-to-large":
        group_counts[2],
}

total_gt = sum(
    actual_counts.values()
)

for name in GROUP_NAMES:
    print(
        f"{name:18s}: "
        f"{actual_counts[name]}"
    )

print(
    f"{'Total':18s}: "
    f"{total_gt}"
)

print(
    f"{'Background images':18s}: "
    f"{background_images}"
)

# ------------------------------------------------------------
# 严格检查是否与论文固定 test 完全一致
# ------------------------------------------------------------

if total_gt != EXPECTED_TOTAL:
    raise RuntimeError(
        f"GT 总数错误："
        f"得到 {total_gt}，"
        f"论文固定 test 应为 "
        f"{EXPECTED_TOTAL}。"
    )

for name in GROUP_NAMES:

    if (
        actual_counts[name]
        != EXPECTED_COUNTS[name]
    ):

        raise RuntimeError(
            f"{name} GT 数量不一致："
            f"得到 "
            f"{actual_counts[name]}，"
            f"论文应为 "
            f"{EXPECTED_COUNTS[name]}。"
            f"\n停止 scale-wise evaluation，"
            f"避免使用错误分组。"
        )

print(
    "\n[OK] Fixed-test GT counts "
    "exactly match the manuscript."
)


# ============================================================
# I. 预测并缓存
# ============================================================


def run_predictions(
    model_name,
    weight,
):
    import gc

    cache_path = (
        PRED_CACHE
        / f"{model_name}.pkl"
    )

    weight_mtime = (
        Path(weight)
        .stat()
        .st_mtime
    )

    # --------------------------------------------------------
    # 如果已经有完整缓存，直接使用
    # --------------------------------------------------------
    if cache_path.exists():

        try:
            with cache_path.open(
                "rb"
            ) as f:

                cache = pickle.load(f)

            if (
                cache.get("weight")
                == str(weight)
                and abs(
                    cache.get(
                        "mtime",
                        -1
                    )
                    - weight_mtime
                ) < 1e-6
                and len(
                    cache.get(
                        "predictions",
                        {}
                    )
                )
                == len(test_images)
            ):

                print(
                    f"\n[CACHE] "
                    f"Loaded complete predictions: "
                    f"{cache_path}"
                )

                return cache[
                    "predictions"
                ]

        except Exception:

            print(
                f"[CACHE] Invalid cache, "
                f"recomputing: "
                f"{cache_path}"
            )

    # --------------------------------------------------------
    # 清显存
    # --------------------------------------------------------
    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"PREDICTION PER-IMAGE: "
        f"{model_name}"
    )

    print(
        "=" * 100
    )

    model = YOLO(
        str(weight)
    )

    predictions = {}

    # --------------------------------------------------------
    # 核心修复：
    #
    # 不再把 572 张图片组成 list
    # 一次送给 model.predict。
    #
    # 每次只推理一张图片。
    # batch 固定为 1。
    # --------------------------------------------------------

    with torch.inference_mode():

        for count, img_path in enumerate(
            test_images,
            1
        ):

            result_list = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=NMS_IOU,
                max_det=MAX_DET,
                augment=False,
                rect=False,
                half=False,
                device=DEVICE,
                batch=1,
                stream=False,
                verbose=False,
            )

            if len(result_list) != 1:

                raise RuntimeError(
                    f"Unexpected result count "
                    f"for {img_path}: "
                    f"{len(result_list)}"
                )

            result = result_list[0]

            result_path = str(
                Path(
                    result.path
                ).resolve()
            )

            if (
                result.boxes is None
                or len(
                    result.boxes
                ) == 0
            ):

                boxes = np.zeros(
                    (0, 4),
                    dtype=np.float32
                )

                confs = np.zeros(
                    (0,),
                    dtype=np.float32
                )

            else:

                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

                confs = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

            predictions[
                result_path
            ] = {
                "boxes": boxes,
                "conf": confs,
            }

            del result
            del result_list

            # 定期释放 CUDA cache
            if (
                count % 50 == 0
            ):

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (
                count % 50 == 0
                or count
                == len(test_images)
            ):

                if torch.cuda.is_available():

                    allocated = (
                        torch.cuda
                        .memory_allocated()
                        / 1024**3
                    )

                    reserved = (
                        torch.cuda
                        .memory_reserved()
                        / 1024**3
                    )

                    memory_text = (
                        f" | CUDA "
                        f"allocated="
                        f"{allocated:.2f}G"
                        f", reserved="
                        f"{reserved:.2f}G"
                    )

                else:

                    memory_text = ""

                print(
                    f"{model_name}: "
                    f"{count}/"
                    f"{len(test_images)}"
                    f"{memory_text}"
                )

    # --------------------------------------------------------
    # 完整性检查
    # --------------------------------------------------------

    if len(
        predictions
    ) != len(
        test_images
    ):

        raise RuntimeError(
            f"{model_name}: "
            f"prediction count "
            f"{len(predictions)} "
            f"!= "
            f"{len(test_images)}"
        )

    missing = []

    for img_path in test_images:

        key = str(
            img_path.resolve()
        )

        if key not in predictions:

            missing.append(
                key
            )

    if missing:

        raise RuntimeError(
            f"{model_name}: "
            f"{len(missing)} "
            f"images missing."
        )

    # --------------------------------------------------------
    # 缓存到 CPU pickle
    # --------------------------------------------------------

    with cache_path.open(
        "wb"
    ) as f:

        pickle.dump(
            {
                "weight":
                    str(weight),

                "mtime":
                    weight_mtime,

                "predictions":
                    predictions,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"[CACHE SAVED] "
        f"{cache_path}"
    )

    # --------------------------------------------------------
    # 删除模型
    # --------------------------------------------------------

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    return predictions



baseline_predictions = (
    run_predictions(
        "baseline_P345",
        BASELINE_PT,
    )
)

h11_predictions = (
    run_predictions(
        "h11_P234",
        H11_PT,
    )
)


# ============================================================
# J. COCO-style 101-point interpolated AP
# ============================================================

def ap_101_point(
    precision,
    recall
):
    if (
        len(precision) == 0
        or len(recall) == 0
    ):
        return 0.0

    recall_points = np.linspace(
        0.0,
        1.0,
        101
    )

    interp = []

    for r in recall_points:

        mask = (
            recall >= r
        )

        if np.any(mask):
            interp.append(
                float(
                    np.max(
                        precision[
                            mask
                        ]
                    )
                )
            )
        else:
            interp.append(
                0.0
            )

    return float(
        np.mean(interp)
    )


# ============================================================
# K. scale-conditioned AP
#
# 关键：
# - target scale GT 参与 TP/FN
# - 与其它尺度 GT 正确重叠的 prediction 被 ignore
# - 不将其它尺度真实目标错误地作为 FP
# ============================================================

def evaluate_one_group(
    predictions,
    group_id,
):
    n_gt = int(
        group_counts[
            group_id
        ]
    )

    if n_gt == 0:
        return (
            0.0,
            0.0
        )

    ap_by_iou = []

    for threshold in IOU_THRESHOLDS:

        records = []

        for img_path in test_images:

            key = str(
                img_path.resolve()
            )

            gt_boxes = (
                gt_db[key][
                    "boxes"
                ]
            )

            gt_groups = (
                gt_db[key][
                    "groups"
                ]
            )

            pred_boxes = (
                predictions[key][
                    "boxes"
                ]
            )

            pred_conf = (
                predictions[key][
                    "conf"
                ]
            )

            if len(pred_boxes) == 0:
                continue

            target_idx = np.where(
                gt_groups
                == group_id
            )[0]

            other_idx = np.where(
                gt_groups
                != group_id
            )[0]

            # 每张图片内部先按 conf 降序
            order = np.argsort(
                -pred_conf
            )

            matched_target = set()

            for pi in order:

                box = pred_boxes[
                    pi
                ]

                conf = float(
                    pred_conf[
                        pi
                    ]
                )

                # ----------------------------------------
                # 先检查 target scale GT
                # ----------------------------------------
                if len(target_idx) > 0:

                    ious_target = (
                        iou_one_to_many(
                            box,
                            gt_boxes[
                                target_idx
                            ]
                        )
                    )

                    best_local = int(
                        np.argmax(
                            ious_target
                        )
                    )

                    best_iou = float(
                        ious_target[
                            best_local
                        ]
                    )

                    best_gt_index = int(
                        target_idx[
                            best_local
                        ]
                    )

                    if best_iou >= threshold:

                        if (
                            best_gt_index
                            not in
                            matched_target
                        ):

                            matched_target.add(
                                best_gt_index
                            )

                            # TP
                            records.append(
                                (
                                    conf,
                                    1,
                                    0,
                                )
                            )

                        else:

                            # duplicate prediction -> FP
                            records.append(
                                (
                                    conf,
                                    0,
                                    1,
                                )
                            )

                        continue

                # ----------------------------------------
                # 如果是其它尺度真实 GT：
                # ignore，而不是 FP
                # ----------------------------------------
                if len(other_idx) > 0:

                    ious_other = (
                        iou_one_to_many(
                            box,
                            gt_boxes[
                                other_idx
                            ]
                        )
                    )

                    if (
                        len(ious_other) > 0
                        and float(
                            np.max(
                                ious_other
                            )
                        )
                        >= threshold
                    ):

                        # ignore
                        continue

                # 无法匹配任何 GT -> FP
                records.append(
                    (
                        conf,
                        0,
                        1,
                    )
                )

        if len(records) == 0:

            ap_by_iou.append(
                0.0
            )

            continue

        # 全数据集按 confidence 排序
        records.sort(
            key=lambda x:
                x[0],
            reverse=True
        )

        tp = np.asarray(
            [
                x[1]
                for x in records
            ],
            dtype=np.float64
        )

        fp = np.asarray(
            [
                x[2]
                for x in records
            ],
            dtype=np.float64
        )

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = (
            tp_cum
            / max(
                n_gt,
                1
            )
        )

        precision = (
            tp_cum
            /
            np.maximum(
                tp_cum
                + fp_cum,
                1e-12
            )
        )

        ap = ap_101_point(
            precision,
            recall
        )

        ap_by_iou.append(
            ap
        )

    ap50 = float(
        ap_by_iou[0]
    )

    ap50_95 = float(
        np.mean(
            ap_by_iou
        )
    )

    return (
        ap50,
        ap50_95
    )


# ============================================================
# L. 计算两模型 scale-conditioned AP
# ============================================================

SCALE_CSV = (
    OUT
    / "scale_conditioned_baseline_vs_h11.csv"
)

DELTA_CSV = (
    OUT
    / "scale_conditioned_delta.csv"
)

models_for_scale = [
    (
        "YOLOv12n_Baseline",
        "P3/P4/P5",
        baseline_predictions,
    ),
    (
        "H11_Reallocation",
        "P2/P3/P4",
        h11_predictions,
    ),
]

scale_rows = []

for (
    model_name,
    hierarchy,
    preds,
) in models_for_scale:

    print("\n" + "=" * 100)
    print(
        "SCALE-CONDITIONED AP:",
        model_name
    )
    print("=" * 100)

    for group_id, group_name in enumerate(
        GROUP_NAMES
    ):

        ap50, ap5095 = (
            evaluate_one_group(
                preds,
                group_id
            )
        )

        row = {
            "Model":
                model_name,

            "Hierarchy":
                hierarchy,

            "Scale_Group":
                group_name,

            "GT_Count":
                group_counts[
                    group_id
                ],

            "AP50":
                ap50,

            "AP50_95":
                ap5095,
        }

        scale_rows.append(
            row
        )

        print(
            f"{group_name:18s} | "
            f"GT={row['GT_Count']:4d} | "
            f"AP50={ap50:.4f} | "
            f"AP50:95={ap5095:.4f}"
        )


with SCALE_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Hierarchy",
            "Scale_Group",
            "GT_Count",
            "AP50",
            "AP50_95",
        ]
    )

    writer.writeheader()
    writer.writerows(
        scale_rows
    )


# ============================================================
# M. H11 - Baseline 差值
# ============================================================

baseline_scale = {
    r["Scale_Group"]: r
    for r in scale_rows
    if r["Model"]
    == "YOLOv12n_Baseline"
}

h11_scale = {
    r["Scale_Group"]: r
    for r in scale_rows
    if r["Model"]
    == "H11_Reallocation"
}

delta_rows = []

for name in GROUP_NAMES:

    b = baseline_scale[
        name
    ]

    h = h11_scale[
        name
    ]

    delta_rows.append({
        "Scale_Group":
            name,

        "GT_Count":
            b["GT_Count"],

        "Baseline_AP50":
            b["AP50"],

        "H11_AP50":
            h["AP50"],

        "Delta_AP50":
            h["AP50"]
            - b["AP50"],

        "Baseline_AP50_95":
            b["AP50_95"],

        "H11_AP50_95":
            h["AP50_95"],

        "Delta_AP50_95":
            h["AP50_95"]
            - b["AP50_95"],
    })


with DELTA_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            delta_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        delta_rows
    )


# ============================================================
# N. 最终终端汇总
# ============================================================

print("\n" + "=" * 100)
print("FIXED-TEST OVERALL RESULT")
print("=" * 100)

for r in overall_rows:

    print(
        f"{r['Model']:22s} | "
        f"{r['Hierarchy']:10s} | "
        f"P={r['Precision']:.4f} | "
        f"R={r['Recall']:.4f} | "
        f"AP50={r['AP50']:.4f} | "
        f"AP50:95={r['AP50_95']:.4f}"
    )


print("\n" + "=" * 100)
print("SCALE-WISE DELTA: H11 - BASELINE")
print("=" * 100)

for r in delta_rows:

    print(
        f"{r['Scale_Group']:18s} | "
        f"GT={r['GT_Count']:4d} | "
        f"Delta AP50="
        f"{r['Delta_AP50']:+.4f} | "
        f"Delta AP50:95="
        f"{r['Delta_AP50_95']:+.4f}"
    )


print("\n" + "=" * 100)
print("INTERPRETATION CHECK")
print("=" * 100)

tiny_delta = delta_rows[0][
    "Delta_AP50"
]

small_delta = delta_rows[1][
    "Delta_AP50"
]

medium_delta = delta_rows[2][
    "Delta_AP50"
]

if (
    tiny_delta > 0
    and small_delta > 0
    and medium_delta < 0
):
    print(
        "[PATTERN] "
        "tiny ↑, small ↑, "
        "medium-to-large ↓"
    )

    print(
        "This is consistent with "
        "prediction capacity shifting "
        "toward finer scales."
    )

elif (
    tiny_delta > 0
    or small_delta > 0
):
    print(
        "[PATTERN] "
        "Partial fine-scale improvement."
    )

    print(
        "Interpret conservatively; "
        "the scale-selective response "
        "is not fully consistent "
        "across both fine-scale groups."
    )

else:
    print(
        "[PATTERN] "
        "No fine-scale AP50 improvement."
    )

    print(
        "Do NOT claim that hierarchy "
        "reallocation alone improves "
        "tiny/small detection."
    )


print("\nFILES SAVED:")

print(
    "1.",
    OVERALL_CSV
)

print(
    "2.",
    SCALE_CSV
)

print(
    "3.",
    DELTA_CSV
)

print(
    "4.",
    PRED_CACHE
)

print("\nDONE.")

