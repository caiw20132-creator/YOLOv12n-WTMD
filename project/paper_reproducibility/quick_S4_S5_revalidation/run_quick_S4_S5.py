from pathlib import Path
import csv
import gc
import hashlib
import shutil
import statistics
import time

import cv2
import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths


# ============================================================
# 0. Configuration
# ============================================================

ROOT = Path(r"D:\YOLO-WTB")

DATA_YAML = ROOT / "data" / "wtb_damage_4021_1151.yaml"

BASELINE_PT = Path(
    r"D:\YOLO-WTB\runs_m4_schemeA_official_8x3"
    r"\train\Baseline_YOLOv12n_seed0\weights\best.pt"
)

WTMD_PT = Path(
    r"D:\YOLO-WTB\runs_wtb"
    r"\FINAL_allin_960_stageB\weights\best.pt"
)

OUT = (
    ROOT
    / "paper_reproducibility"
    / "quick_S4_S5_revalidation"
)

ULTRA_OUT = OUT / "ultralytics_runs"
TEMP_ROOT = OUT / "_temporary_degraded_test"

OUT.mkdir(parents=True, exist_ok=True)
ULTRA_OUT.mkdir(parents=True, exist_ok=True)

DEVICE = 0
BATCH = 6
WORKERS = 0

CONF = 0.001
IOU = 0.50
MAX_DET = 10000

EXPECTED_TEST_IMAGES = 572

# Deterministic Gaussian-noise generation
NOISE_SEED = 20261005

DEGRADATIONS = [
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "low_brightness",
    "low_contrast",
    "jpeg_compression",
]

RESOLUTIONS = [
    640,
    800,
    960,
    1280,
]


# ============================================================
# 1. Utility functions
# ============================================================

def stage(text):
    print("\n" + "=" * 88, flush=True)
    print(text, flush=True)
    print("=" * 88, flush=True)


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def clear_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inspect_model(model, checkpoint):
    net = model.model

    params_m = (
        sum(p.numel() for p in net.parameters())
        / 1.0e6
    )

    stride = getattr(net, "stride", None)

    if stride is None:
        strides = []
    else:
        try:
            strides = [
                int(round(float(x)))
                for x in stride.detach().cpu().tolist()
            ]
        except Exception:
            strides = [
                int(round(float(x)))
                for x in stride
            ]

    print(f"Checkpoint : {checkpoint}", flush=True)
    print(f"Params     : {params_m:.6f} M", flush=True)
    print(f"Strides    : {strides}", flush=True)

    return {
        "params_m": params_m,
        "strides": strides,
    }


# ============================================================
# 2. Resolve fixed-test images
# ============================================================

def collect_images(entry):
    valid_ext = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    output = []

    if isinstance(entry, (list, tuple)):
        for item in entry:
            output.extend(collect_images(item))
        return output

    p = Path(str(entry))

    if p.is_dir():
        output.extend(
            q
            for q in p.rglob("*")
            if q.is_file()
            and q.suffix.lower() in valid_ext
        )

    elif p.is_file() and p.suffix.lower() == ".txt":
        for line in p.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            q = Path(line)

            if not q.is_absolute():
                q = (p.parent / q).resolve()

            if q.exists():
                output.append(q)

    elif p.is_file():
        if p.suffix.lower() in valid_ext:
            output.append(p)

    return sorted(
        set(output),
        key=lambda x: str(x).lower()
    )


stage("[1/8] Resolving WTMD fixed-test subset")

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

data_info = check_det_dataset(
    str(DATA_YAML),
    autodownload=False,
)

test_entry = data_info.get("test")

if test_entry is None:
    raise RuntimeError(
        "Dataset YAML contains no test split."
    )

test_images = collect_images(test_entry)

print(
    f"Fixed-test images: {len(test_images)}",
    flush=True,
)

if len(test_images) != EXPECTED_TEST_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_TEST_IMAGES} fixed-test images, "
        f"but found {len(test_images)}. STOPPING."
    )

names = data_info.get(
    "names",
    {0: "damage"},
)


# ============================================================
# 3. Load fixed checkpoint pair
# ============================================================

stage("[2/8] Loading fixed seed-0 checkpoint pair")

for path in (
    BASELINE_PT,
    WTMD_PT,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

baseline_model = YOLO(str(BASELINE_PT))
wtmd_model = YOLO(str(WTMD_PT))

baseline_info = inspect_model(
    baseline_model,
    BASELINE_PT,
)

wtmd_info = inspect_model(
    wtmd_model,
    WTMD_PT,
)

# Architecture sanity checks
if baseline_info["strides"] != [8, 16, 32]:
    raise RuntimeError(
        "Baseline stride check failed. "
        f"Expected [8,16,32], got "
        f"{baseline_info['strides']}"
    )

if not (
    2.45
    <= baseline_info["params_m"]
    <= 2.65
):
    raise RuntimeError(
        "Baseline parameter count is inconsistent "
        "with YOLOv12n."
    )

if wtmd_info["strides"] != [4, 8, 16]:
    raise RuntimeError(
        "WTMD stride check failed. "
        f"Expected [4,8,16], got "
        f"{wtmd_info['strides']}"
    )

if not (
    1.42
    <= wtmd_info["params_m"]
    <= 1.52
):
    raise RuntimeError(
        "WTMD parameter count is inconsistent "
        "with YOLOv12n-WTMD."
    )


# ============================================================
# 4. Official validation helper
# ============================================================

def run_val(
    model,
    data,
    imgsz,
    name,
):
    print(
        f"\n[VAL] {name} | imgsz={imgsz}",
        flush=True,
    )

    result = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=BATCH,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=DEVICE,
        workers=WORKERS,
        augment=False,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(ULTRA_OUT),
        name=name,
        exist_ok=True,
    )

    metrics = {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "ap50": float(result.box.map50),
        "ap50_95": float(result.box.map),
    }

    print(
        "  "
        f"P={metrics['precision']:.6f} | "
        f"R={metrics['recall']:.6f} | "
        f"AP50={metrics['ap50']:.6f} | "
        f"AP50:95={metrics['ap50_95']:.6f}",
        flush=True,
    )

    clear_cuda()

    return metrics


# ============================================================
# 5. CLEAN 960 sanity test
# ============================================================

stage("[3/8] Fast clean-condition sanity validation")

clean = {}

clean["YOLOv12n"] = run_val(
    baseline_model,
    DATA_YAML,
    960,
    "clean_YOLOv12n_960",
)

clean["YOLOv12n-WTMD"] = run_val(
    wtmd_model,
    DATA_YAML,
    960,
    "clean_YOLOv12n_WTMD_960",
)

print("\nClean-condition check:", flush=True)

for model_name, m in clean.items():
    print(
        f"{model_name:<18} "
        f"AP50={m['ap50']:.6f} | "
        f"AP50:95={m['ap50_95']:.6f} | "
        f"Recall={m['recall']:.6f}",
        flush=True,
    )

# Stop early if clearly wrong checkpoints.
if (
    clean["YOLOv12n"]["ap50"] < 0.60
    or
    clean["YOLOv12n-WTMD"]["ap50"] < 0.60
):
    raise RuntimeError(
        "\nCLEAN AP50 IS BELOW 0.60.\n"
        "This checkpoint pair is not compatible with "
        "the current paper's primary model family.\n"
        "Full S4/S5 evaluation has been stopped to "
        "avoid wasting time."
    )

print(
    "\nPASS: checkpoint pair is suitable for "
    "the quick operating-boundary rerun.",
    flush=True,
)


# ============================================================
# 6. Degradation generation
# ============================================================

def degrade_image(
    image,
    condition,
    index,
):
    if condition == "gaussian_blur":
        return cv2.GaussianBlur(
            image,
            (5, 5),
            1.5,
        )

    if condition == "motion_blur":
        # Explicit reproducible 9-pixel horizontal
        # linear motion-blur kernel.
        kernel = np.zeros(
            (9, 9),
            dtype=np.float32,
        )

        kernel[4, :] = 1.0 / 9.0

        return cv2.filter2D(
            image,
            -1,
            kernel,
        )

    if condition == "gaussian_noise":
        rng = np.random.default_rng(
            NOISE_SEED + index
        )

        noise = rng.normal(
            0.0,
            15.0,
            image.shape,
        ).astype(np.float32)

        output = (
            image.astype(np.float32)
            + noise
        )

        return np.clip(
            output,
            0,
            255,
        ).astype(np.uint8)

    if condition == "low_brightness":
        output = (
            image.astype(np.float32)
            * 0.65
        )

        return np.clip(
            output,
            0,
            255,
        ).astype(np.uint8)

    if condition == "low_contrast":
        output = (
            (
                image.astype(np.float32)
                - 127.5
            )
            * 0.65
            + 127.5
        )

        return np.clip(
            output,
            0,
            255,
        ).astype(np.uint8)

    if condition == "jpeg_compression":
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                40,
            ],
        )

        if not ok:
            raise RuntimeError(
                "JPEG encoding failed."
            )

        return cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

    raise ValueError(
        f"Unknown degradation: {condition}"
    )


def build_degraded_dataset(condition):
    if TEMP_ROOT.exists():
        shutil.rmtree(
            TEMP_ROOT,
            ignore_errors=True,
        )

    images_dir = (
        TEMP_ROOT
        / "images"
        / "test"
    )

    labels_dir = (
        TEMP_ROOT
        / "labels"
        / "test"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nGenerating {condition} dataset...",
        flush=True,
    )

    for index, source in enumerate(
        test_images,
        start=1,
    ):
        image = cv2.imread(
            str(source),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {source}"
            )

        output = degrade_image(
            image,
            condition,
            index,
        )

        stem = (
            f"{index:06d}_"
            f"{source.stem}"
        )

        # PNG avoids introducing unintended compression
        # into non-JPEG degradation conditions.
        suffix = (
            ".jpg"
            if condition == "jpeg_compression"
            else ".png"
        )

        destination = (
            images_dir
            / f"{stem}{suffix}"
        )

        if suffix == ".jpg":
            ok = cv2.imwrite(
                str(destination),
                output,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    100,
                ],
            )
        else:
            ok = cv2.imwrite(
                str(destination),
                output,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )

        if not ok:
            raise RuntimeError(
                f"Cannot write: {destination}"
            )

        source_label = Path(
            img2label_paths(
                [str(source)]
            )[0]
        )

        if source_label.exists():
            destination_label = (
                labels_dir
                / f"{stem}.txt"
            )

            shutil.copy2(
                source_label,
                destination_label,
            )

        if (
            index % 100 == 0
            or index == len(test_images)
        ):
            print(
                f"  {index}/{len(test_images)}",
                flush=True,
            )

    yaml_path = (
        TEMP_ROOT
        / f"{condition}.yaml"
    )

    payload = {
        "path": str(TEMP_ROOT),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": names,
    }

    yaml_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    return yaml_path


# ============================================================
# 7. Regenerate Supplementary Table S4
# ============================================================

stage("[4/8] Regenerating S4 image-degradation sensitivity")

s4_metrics = {
    "YOLOv12n": {
        "Clean": clean["YOLOv12n"],
    },
    "YOLOv12n-WTMD": {
        "Clean": clean["YOLOv12n-WTMD"],
    },
}

display_names = {
    "gaussian_blur": "Gaussian blur",
    "motion_blur": "Motion blur",
    "gaussian_noise": "Gaussian noise",
    "low_brightness": "Low brightness",
    "low_contrast": "Low contrast",
    "jpeg_compression": "JPEG compression",
}

for condition in DEGRADATIONS:
    data_yaml = build_degraded_dataset(
        condition
    )

    label = display_names[condition]

    s4_metrics["YOLOv12n"][label] = (
        run_val(
            baseline_model,
            data_yaml,
            960,
            f"S4_{condition}_baseline",
        )
    )

    s4_metrics["YOLOv12n-WTMD"][label] = (
        run_val(
            wtmd_model,
            data_yaml,
            960,
            f"S4_{condition}_wtmd",
        )
    )

    shutil.rmtree(
        TEMP_ROOT,
        ignore_errors=True,
    )


s4_csv = OUT / "S4_REGENERATED.csv"

with s4_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    fields = [
        "Model",
        "Image degradation",
        "Precision",
        "Recall",
        "AP50",
        "AP50:95",
        "AP50 drop",
        "AP50 retention (%)",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for model_name in [
        "YOLOv12n",
        "YOLOv12n-WTMD",
    ]:
        clean_ap50 = (
            s4_metrics[model_name]
            ["Clean"]
            ["ap50"]
        )

        for condition_name, m in (
            s4_metrics[model_name].items()
        ):
            drop = (
                clean_ap50
                - m["ap50"]
            )

            retention = (
                100.0
                * m["ap50"]
                / clean_ap50
            )

            writer.writerow({
                "Model": model_name,
                "Image degradation":
                    condition_name,
                "Precision":
                    f"{m['precision']:.6f}",
                "Recall":
                    f"{m['recall']:.6f}",
                "AP50":
                    f"{m['ap50']:.6f}",
                "AP50:95":
                    f"{m['ap50_95']:.6f}",
                "AP50 drop":
                    f"{drop:.6f}",
                "AP50 retention (%)":
                    f"{retention:.4f}",
            })


# ============================================================
# 8. Tensor-forward benchmark helper
# ============================================================

def benchmark_forward(
    model,
    resolution,
    warmup=30,
    repeats=100,
):
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    net = (
        model.model
        .to(device)
        .float()
        .eval()
    )

    tensor = torch.zeros(
        1,
        3,
        resolution,
        resolution,
        device=device,
        dtype=torch.float32,
    )

    times = []

    with torch.inference_mode():
        for _ in range(warmup):
            _ = net(tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = net(tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            times.append(elapsed)

    mean_ms = statistics.mean(times)

    std_ms = (
        statistics.stdev(times)
        if len(times) > 1
        else 0.0
    )

    fps = 1000.0 / mean_ms

    del tensor
    clear_cuda()

    return mean_ms, std_ms, fps


# ============================================================
# 9. Regenerate Supplementary Table S5
# ============================================================

stage("[5/8] Regenerating S5 resolution sensitivity")

s5_metrics = {
    "YOLOv12n": {},
    "YOLOv12n-WTMD": {},
}

# Reuse already computed clean 960 metrics.
s5_metrics["YOLOv12n"][960] = (
    clean["YOLOv12n"]
)

s5_metrics["YOLOv12n-WTMD"][960] = (
    clean["YOLOv12n-WTMD"]
)

for resolution in [
    640,
    800,
    1280,
]:
    s5_metrics["YOLOv12n"][resolution] = (
        run_val(
            baseline_model,
            DATA_YAML,
            resolution,
            f"S5_{resolution}_baseline",
        )
    )

    s5_metrics["YOLOv12n-WTMD"][resolution] = (
        run_val(
            wtmd_model,
            DATA_YAML,
            resolution,
            f"S5_{resolution}_wtmd",
        )
    )


stage("[6/8] Tensor-forward timing for S5")

latency = {
    "YOLOv12n": {},
    "YOLOv12n-WTMD": {},
}

for resolution in RESOLUTIONS:
    print(
        f"\nBenchmark YOLOv12n "
        f"{resolution}x{resolution}",
        flush=True,
    )

    latency["YOLOv12n"][resolution] = (
        benchmark_forward(
            baseline_model,
            resolution,
        )
    )

    print(
        f"Benchmark YOLOv12n-WTMD "
        f"{resolution}x{resolution}",
        flush=True,
    )

    latency[
        "YOLOv12n-WTMD"
    ][resolution] = (
        benchmark_forward(
            wtmd_model,
            resolution,
        )
    )


s5_csv = OUT / "S5_REGENERATED.csv"

with s5_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    fields = [
        "Model",
        "Input resolution",
        "Precision",
        "Recall",
        "AP50",
        "AP50:95",
        "Latency mean (ms)",
        "Latency std (ms)",
        "Equivalent tensor-forward FPS",
        "Delta AP50 vs 960",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for model_name in [
        "YOLOv12n",
        "YOLOv12n-WTMD",
    ]:
        reference = (
            s5_metrics[model_name]
            [960]
            ["ap50"]
        )

        for resolution in RESOLUTIONS:
            m = (
                s5_metrics[model_name]
                [resolution]
            )

            mean_ms, std_ms, fps = (
                latency[model_name]
                [resolution]
            )

            writer.writerow({
                "Model": model_name,
                "Input resolution":
                    f"{resolution}x{resolution}",
                "Precision":
                    f"{m['precision']:.6f}",
                "Recall":
                    f"{m['recall']:.6f}",
                "AP50":
                    f"{m['ap50']:.6f}",
                "AP50:95":
                    f"{m['ap50_95']:.6f}",
                "Latency mean (ms)":
                    f"{mean_ms:.6f}",
                "Latency std (ms)":
                    f"{std_ms:.6f}",
                "Equivalent tensor-forward FPS":
                    f"{fps:.6f}",
                "Delta AP50 vs 960":
                    f"{m['ap50'] - reference:+.6f}",
            })


# ============================================================
# 10. Exact checkpoint provenance file
# ============================================================

stage("[7/8] Writing reproducibility record")

record = OUT / "CHECKPOINTS_USED.txt"

record.write_text(
    "\n".join([
        "S4 / S5 QUICK REVALIDATION",
        "============================================",
        "",
        "Baseline checkpoint:",
        str(BASELINE_PT),
        "",
        "Baseline SHA256:",
        sha256_file(BASELINE_PT),
        "",
        "Baseline parameters:",
        f"{baseline_info['params_m']:.6f} M",
        "",
        "Baseline strides:",
        str(baseline_info["strides"]),
        "",
        "YOLOv12n-WTMD checkpoint:",
        str(WTMD_PT),
        "",
        "YOLOv12n-WTMD SHA256:",
        sha256_file(WTMD_PT),
        "",
        "YOLOv12n-WTMD parameters:",
        f"{wtmd_info['params_m']:.6f} M",
        "",
        "YOLOv12n-WTMD strides:",
        str(wtmd_info["strides"]),
        "",
        "Dataset:",
        str(DATA_YAML),
        "",
        f"Fixed-test images: {len(test_images)}",
        "",
        "Evaluation:",
        "conf=0.001",
        "NMS IoU=0.50",
        "max_det=10000",
        "augment=False",
        "batch=6",
        "",
        "Degradation protocol:",
        "Gaussian blur: kernel=5, sigma=1.5",
        "Motion blur: 9-pixel horizontal linear kernel",
        "Gaussian noise: sigma=15, deterministic",
        "Low brightness: factor=0.65",
        "Low contrast: factor=0.65 around 127.5",
        "JPEG compression: quality=40",
        "",
        "These S4/S5 results are fixed-pair",
        "supplementary operating-boundary results,",
        "not three-seed averages.",
    ]),
    encoding="utf-8",
)


# ============================================================
# 11. Compact human-readable summary
# ============================================================

summary = OUT / "QUICK_VERIFY_SUMMARY.txt"

lines = []

lines.append(
    "============================================================"
)
lines.append(
    "QUICK S4 / S5 REVALIDATION SUMMARY"
)
lines.append(
    "============================================================"
)
lines.append("")

lines.append(
    f"Baseline: {BASELINE_PT}"
)
lines.append(
    f"WTMD:     {WTMD_PT}"
)
lines.append("")

lines.append(
    "CLEAN 960:"
)

for model_name in [
    "YOLOv12n",
    "YOLOv12n-WTMD",
]:
    m = clean[model_name]

    lines.append(
        f"{model_name}: "
        f"P={m['precision']:.4f}, "
        f"R={m['recall']:.4f}, "
        f"AP50={m['ap50']:.4f}, "
        f"AP50:95={m['ap50_95']:.4f}"
    )

lines.append("")
lines.append(
    f"S4 CSV: {s4_csv}"
)
lines.append(
    f"S5 CSV: {s5_csv}"
)
lines.append(
    f"Checkpoint record: {record}"
)

summary.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ============================================================
# Done
# ============================================================

stage("[8/8] COMPLETE")

print(
    f"S4 result: {s4_csv}",
    flush=True,
)

print(
    f"S5 result: {s5_csv}",
    flush=True,
)

print(
    f"Checkpoint record: {record}",
    flush=True,
)

print(
    f"Summary: {summary}",
    flush=True,
)

print(
    "\nPlease send QUICK_VERIFY_SUMMARY.txt, "
    "S4_REGENERATED.csv and S5_REGENERATED.csv "
    "back to ChatGPT.",
    flush=True,
)

