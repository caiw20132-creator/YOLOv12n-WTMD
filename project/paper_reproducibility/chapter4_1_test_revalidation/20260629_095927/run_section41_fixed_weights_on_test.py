import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml


IMGSZ = 960
BATCH = 1
CONF = 0.001
IOU = 0.50
MAX_DET = 10000
DEVICE = 0
WORKERS = 0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def log(message: str):
    print(message, flush=True)


def sha256_with_progress(path: Path, label: str) -> str:
    total = max(path.stat().st_size, 1)
    processed = 0
    next_report = 0.25
    digest = hashlib.sha256()

    log(f"[SHA256] {label}")
    log(f"         path={path}")

    with path.open("rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)
            processed += len(chunk)

            ratio = processed / total

            if ratio >= next_report:
                log(f"         progress={ratio * 100:.0f}%")
                next_report += 0.25

    return digest.hexdigest()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(str(value))

    if path.is_absolute():
        return path

    return (base / path).resolve()


def count_test_images(data_yaml: Path):
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise RuntimeError(f"Invalid dataset YAML: {data_yaml}")

    if "test" not in config or not config["test"]:
        raise RuntimeError(
            "The dataset YAML does not define a non-empty 'test' entry. "
            "Section 4.1 must use the fixed WTMD test split."
        )

    yaml_dir = data_yaml.parent
    root_dir = yaml_dir

    if config.get("path"):
        root_dir = resolve_path(yaml_dir, config["path"])

    test_path = resolve_path(root_dir, config["test"])

    if test_path.is_file():
        image_paths = []

        for line in test_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            image_path = Path(line)

            if not image_path.is_absolute():
                image_path = (test_path.parent / image_path).resolve()

            image_paths.append(image_path)

    elif test_path.is_dir():
        image_paths = sorted(
            [
                path
                for path in test_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda item: str(item).lower(),
        )

    else:
        raise FileNotFoundError(
            f"Unable to resolve test data from YAML entry: {test_path}"
        )

    return config, test_path, image_paths


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)

    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")

    text = str(image_path)
    text = text.replace("\\images\\", "\\labels\\")
    text = text.replace("/images/", "/labels/")

    return Path(text).with_suffix(".txt")


def count_test_instances(image_paths):
    count = 0

    for image_path in image_paths:
        label_path = label_path_from_image(image_path)

        if not label_path.is_file():
            continue

        for line in label_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            if line.strip():
                count += 1

    return count


def register_sf1sppf(root: Path):
    log("[CUSTOM MODULE] Registering SF1SPPF for the StageB checkpoint...")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        sf1_sppf = importlib.import_module("sf1_sppf")

        from ultralytics.nn import modules, tasks

        registered = []

        for class_name in (
            "SF1SPPF",
            "SoftPoolSPPF",
            "MaxPoolSF1SPPF",
        ):
            if hasattr(sf1_sppf, class_name):
                cls = getattr(sf1_sppf, class_name)

                setattr(tasks, class_name, cls)
                setattr(modules, class_name, cls)

                tasks.__dict__[class_name] = cls
                modules.__dict__[class_name] = cls

                registered.append(class_name)

        if "SF1SPPF" not in registered:
            raise RuntimeError(
                "sf1_sppf.py was imported but SF1SPPF was not found."
            )

        log(
            "[CUSTOM MODULE] Registration completed: "
            + ", ".join(registered)
        )

    except Exception as error:
        raise RuntimeError(
            "Unable to register custom SF1SPPF before loading the "
            f"YOLOv12n-WTMD checkpoint. {type(error).__name__}: {error}"
        )


def get_structure(model):
    params_m = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    ) / 1e6

    detect_layers = [
        module
        for module in model.model.modules()
        if module.__class__.__name__ == "Detect"
    ]

    strides = []

    if detect_layers:
        stride = getattr(detect_layers[-1], "stride", None)

        if stride is not None:
            try:
                strides = [
                    float(value)
                    for value in stride.detach().cpu().tolist()
                ]
            except Exception:
                try:
                    strides = [float(value) for value in stride]
                except Exception:
                    strides = []

    return {
        "params_raw_m": params_m,
        "strides": ",".join(str(int(value)) for value in strides),
    }


def validate_checkpoint(
    label: str,
    checkpoint: Path,
    data_yaml: Path,
    output_dir: Path,
):
    from ultralytics import YOLO

    log("")
    log("=" * 100)
    log(f"[VALIDATE] {label}")
    log(f"[CHECKPOINT] {checkpoint}")
    log(
        "[PROTOCOL] "
        f"split=test | imgsz={IMGSZ} | batch={BATCH} | "
        f"conf={CONF} | iou={IOU} | max_det={MAX_DET} | "
        "augment=False | rect=False | half=False"
    )
    log("=" * 100)

    log(f"[LOAD] Loading {label} checkpoint...")

    detector = YOLO(str(checkpoint))
    structure = get_structure(detector)

    log(
        "[MODEL INFO] "
        f"params(raw)={structure['params_raw_m']:.6f} M | "
        f"strides=[{structure['strides']}]"
    )

    log(
        f"[RUN] Starting test inference for {label}. "
        "Ultralytics batch progress will appear below."
    )

    metrics = detector.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        augment=False,
        rect=False,
        half=False,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        save_json=False,
        project=str(output_dir / "test_runs"),
        name=label,
        exist_ok=True,
        verbose=True,
    )

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "ap50": float(metrics.box.map50),
        "ap50_95": float(metrics.box.map),
        "speed_preprocess_ms": float(
            metrics.speed.get("preprocess", float("nan"))
        ),
        "speed_inference_ms": float(
            metrics.speed.get("inference", float("nan"))
        ),
        "speed_postprocess_ms": float(
            metrics.speed.get("postprocess", float("nan"))
        ),
        "save_dir": str(getattr(metrics, "save_dir", "")),
        **structure,
    }

    log(
        f"[RESULT] {label} | "
        f"P={result['precision']:.6f} | "
        f"R={result['recall']:.6f} | "
        f"AP50={result['ap50']:.6f} | "
        f"AP50:95={result['ap50_95']:.6f}"
    )

    del detector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_csv(path: Path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--wtmd", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_yaml = Path(args.data).resolve()
    baseline_weight = Path(args.baseline).resolve()
    wtmd_weight = Path(args.wtmd).resolve()
    output_dir = Path(args.output).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    log("[PYTHON PHASE 1/5] Verifying input paths...")

    for name, path in (
        ("Project root", root),
        ("Dataset YAML", data_yaml),
        ("YOLOv12n checkpoint", baseline_weight),
        ("YOLOv12n-WTMD checkpoint", wtmd_weight),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

        log(f"    [OK] {name}: {path}")

    log("[PYTHON PHASE 1/5] Completed.")

    log("")
    log("[PYTHON PHASE 2/5] Auditing the fixed WTMD test split...")

    config, test_path, test_images = count_test_images(data_yaml)
    test_instances = count_test_instances(test_images)

    log(f"    Dataset YAML: {data_yaml}")
    log(f"    Test source: {test_path}")
    log(f"    Test images: {len(test_images)}")
    log(f"    Test instances counted from labels: {test_instances}")
    log(f"    Classes (nc): {config.get('nc', '')}")

    if len(test_images) != 572:
        raise RuntimeError(
            "WTMD test-set image count mismatch. "
            f"Expected 572 images but found {len(test_images)}. "
            "Stopped before evaluation to prevent using the wrong split."
        )

    log("    [OK] Test-set count matches the fixed 572-image test split.")
    log("[PYTHON PHASE 2/5] Completed.")

    log("")
    log("[PYTHON PHASE 3/5] Registering model dependencies and hashing checkpoints...")

    register_sf1sppf(root)

    baseline_sha = sha256_with_progress(
        baseline_weight,
        "YOLOv12n",
    )

    wtmd_sha = sha256_with_progress(
        wtmd_weight,
        "YOLOv12n-WTMD",
    )

    log("[PYTHON PHASE 3/5] Completed.")

    log("")
    log("[PYTHON PHASE 4/5] Running fixed test evaluation for 2 checkpoints...")

    models = [
        ("YOLOv12n", baseline_weight, baseline_sha),
        ("YOLOv12n-WTMD", wtmd_weight, wtmd_sha),
    ]

    rows = []

    for index, (label, checkpoint, digest) in enumerate(
        models,
        start=1,
    ):
        log("")
        log(
            f"[MODEL {index}/{len(models)}] "
            f"Beginning test evaluation: {label}"
        )

        result = validate_checkpoint(
            label=label,
            checkpoint=checkpoint,
            data_yaml=data_yaml,
            output_dir=output_dir,
        )

        rows.append(
            {
                "model": label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "split": "test",
                "test_images": len(test_images),
                "test_instances": test_instances,
                **result,
            }
        )

        log(
            f"[MODEL {index}/{len(models)}] "
            f"Completed test evaluation: {label}"
        )

    baseline_row = rows[0]
    wtmd_row = rows[1]

    delta_row = {
        "model": "YOLOv12n-WTMD minus YOLOv12n",
        "checkpoint": "",
        "checkpoint_sha256": "",
        "split": "test",
        "test_images": len(test_images),
        "test_instances": test_instances,
        "precision": wtmd_row["precision"] - baseline_row["precision"],
        "recall": wtmd_row["recall"] - baseline_row["recall"],
        "ap50": wtmd_row["ap50"] - baseline_row["ap50"],
        "ap50_95": wtmd_row["ap50_95"] - baseline_row["ap50_95"],
        "params_raw_m": (
            wtmd_row["params_raw_m"]
            - baseline_row["params_raw_m"]
        ),
        "strides": "",
        "speed_preprocess_ms": (
            wtmd_row["speed_preprocess_ms"]
            - baseline_row["speed_preprocess_ms"]
        ),
        "speed_inference_ms": (
            wtmd_row["speed_inference_ms"]
            - baseline_row["speed_inference_ms"]
        ),
        "speed_postprocess_ms": (
            wtmd_row["speed_postprocess_ms"]
            - baseline_row["speed_postprocess_ms"]
        ),
        "save_dir": "",
    }

    all_rows = rows + [delta_row]

    summary_path = output_dir / "section4_1_fixed_test_summary.csv"

    fields = [
        "model",
        "checkpoint",
        "checkpoint_sha256",
        "split",
        "test_images",
        "test_instances",
        "precision",
        "recall",
        "ap50",
        "ap50_95",
        "params_raw_m",
        "strides",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
        "save_dir",
    ]

    write_csv(
        summary_path,
        all_rows,
        fields,
    )

    manifest = {
        "purpose": (
            "Section 4.1 fixed-checkpoint evaluation "
            "on the WTMD held-out test split"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "test_source": str(test_path),
        "test_images": len(test_images),
        "test_instances_counted_from_labels": test_instances,
        "protocol": {
            "split": "test",
            "imgsz": IMGSZ,
            "batch": BATCH,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "augment": False,
            "rect": False,
            "half": False,
            "device": DEVICE,
            "workers": WORKERS,
        },
        "results": all_rows,
    }

    manifest_path = output_dir / "section4_1_fixed_test_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("")
    log("[PYTHON PHASE 5/5] Exporting summary and completing run...")
    log("")
    log("=" * 100)
    log("SECTION_4_1_FIXED_TEST_EVALUATION_COMPLETED")
    log(f"SUMMARY_CSV={summary_path}")
    log(f"MANIFEST_JSON={manifest_path}")
    log(
        f"YOLOv12n test AP50={baseline_row['ap50']:.6f} | "
        f"AP50:95={baseline_row['ap50_95']:.6f}"
    )
    log(
        f"YOLOv12n-WTMD test AP50={wtmd_row['ap50']:.6f} | "
        f"AP50:95={wtmd_row['ap50_95']:.6f}"
    )
    log(
        f"Delta test AP50={delta_row['ap50']:+.6f} | "
        f"Delta test AP50:95={delta_row['ap50_95']:+.6f}"
    )
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback.print_exc()
        raise
