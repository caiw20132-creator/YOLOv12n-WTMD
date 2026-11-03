import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml


TEST_IMGSZ = 960
TEST_BATCH = 1
TEST_CONF = 0.001
TEST_IOU = 0.50
TEST_MAX_DET = 10000
TEST_DEVICE = 0
TEST_WORKERS = 0


def log(message: str):
    print(message, flush=True)


def sha256_with_progress(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    total = max(path.stat().st_size, 1)
    completed = 0
    next_report = 0.25

    log(f"[SHA256] {label}")
    log(f"         path={path}")

    with path.open("rb") as file:
        while True:
            block = file.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)
            completed += len(block)

            fraction = completed / total

            if fraction >= next_report:
                log(f"         progress={fraction * 100:.0f}%")
                next_report += 0.25

    return digest.hexdigest()


def normalize_pooling_name(value):
    if isinstance(value, str):
        replacements = {
            "SF1SPPF": "<POOLING_MODULE>",
            "SoftPoolSPPF": "<POOLING_MODULE>",
            "MaxPoolSF1SPPF": "<POOLING_MODULE>",
            "ultralytics.nn.modules.sf1_sppf.SF1SPPF": "<POOLING_MODULE>",
            "wtb_paired_maxpool_module.MaxPoolSF1SPPF": "<POOLING_MODULE>",
        }

        return replacements.get(value, value)

    if isinstance(value, list):
        return [normalize_pooling_name(item) for item in value]

    if isinstance(value, dict):
        return {
            key: normalize_pooling_name(item)
            for key, item in value.items()
        }

    return value


def verify_yaml_pair(soft_yaml: Path, max_yaml: Path):
    log("[YAML CHECK] Verifying that SoftPool and MaxPool YAML files differ only by pooling operator...")

    soft_cfg = yaml.safe_load(
        soft_yaml.read_text(encoding="utf-8")
    )

    max_cfg = yaml.safe_load(
        max_yaml.read_text(encoding="utf-8")
    )

    soft_normalized = normalize_pooling_name(soft_cfg)
    max_normalized = normalize_pooling_name(max_cfg)

    if soft_normalized != max_normalized:
        raise RuntimeError(
            "SoftPool and MaxPool YAML files differ beyond the intended "
            "pooling-module replacement. Training has been stopped."
        )

    log("[YAML CHECK] PASS: only the pooling operator differs.")


def import_module_from_file(module_name: str, source_path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(source_path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot create import specification for {source_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def register_custom_modules(root: Path, maxpool_source: Path):
    log("[CUSTOM MODULE] Registering SoftPool and MaxPool classes...")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        importlib.import_module("sitecustomize")
        log("    [OK] sitecustomize imported.")
    except Exception as error:
        log(f"    [INFO] sitecustomize not loaded: {error}")

    try:
        soft_module = importlib.import_module("sf1_sppf")
    except Exception as error:
        raise RuntimeError(
            f"Unable to import SoftPool module sf1_sppf.py: {error}"
        )

    max_module = import_module_from_file(
        "wtb_paired_maxpool_module",
        maxpool_source,
    )

    from ultralytics.nn import modules, tasks
    from ultralytics.nn.modules import block

    holders = [tasks, modules, block]

    registered = []

    for imported_module in (soft_module, max_module):
        for class_name in (
            "SF1SPPF",
            "SoftPoolSPPF",
            "MaxPoolSF1SPPF",
        ):
            if not hasattr(imported_module, class_name):
                continue

            cls = getattr(imported_module, class_name)

            for holder in holders:
                setattr(holder, class_name, cls)

            registered.append(
                f"{class_name} <- {cls.__module__}"
            )

    if not any("SF1SPPF" in item for item in registered):
        raise RuntimeError(
            "SF1SPPF was not registered. "
            "The StageB SoftPool checkpoint cannot be safely loaded."
        )

    if not any("MaxPoolSF1SPPF" in item for item in registered):
        raise RuntimeError(
            "MaxPoolSF1SPPF was not registered. "
            "The MaxPool control cannot be safely built."
        )

    for item in registered:
        log(f"    [REGISTERED] {item}")

    return registered


def patch_runtime_attributes(model):
    patched = []

    for module in model.modules():
        if module.__class__.__name__ not in {
            "SF1SPPF",
            "SoftPoolSPPF",
            "MaxPoolSF1SPPF",
        }:
            continue

        if hasattr(module, "expected_c1"):
            continue

        try:
            module.expected_c1 = int(
                module.cv1.conv.in_channels
            )

            patched.append(
                module.__class__.__name__
            )

        except Exception:
            pass

    return patched


def inspect_pool_module(detector, expected_class: str):
    found = []

    for module_path, module in detector.model.named_modules():
        if module.__class__.__name__ not in {
            "SF1SPPF",
            "SoftPoolSPPF",
            "MaxPoolSF1SPPF",
        }:
            continue

        cv1 = getattr(module, "cv1", None)
        conv = getattr(cv1, "conv", cv1)

        found.append(
            {
                "module_path": module_path,
                "class_name": module.__class__.__name__,
                "expected_c1": getattr(module, "expected_c1", None),
                "cv1_input_channels": getattr(
                    conv,
                    "in_channels",
                    None,
                ),
            }
        )

    if len(found) != 1:
        raise RuntimeError(
            "Expected exactly one pooling-aggregation module, "
            f"but found {len(found)}: {found}"
        )

    item = found[0]

    if item["class_name"] != expected_class:
        raise RuntimeError(
            f"Expected {expected_class}, but found "
            f"{item['class_name']} at {item['module_path']}."
        )

    if (
        item["expected_c1"] is None
        or item["cv1_input_channels"] is None
        or int(item["expected_c1"])
        != int(item["cv1_input_channels"])
    ):
        raise RuntimeError(
            f"Pooling module channel check failed: {item}"
        )

    return item


def count_split(data_yaml: Path, split_name: str):
    config = yaml.safe_load(
        data_yaml.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise RuntimeError(
            f"Invalid dataset YAML: {data_yaml}"
        )

    if split_name not in config or not config[split_name]:
        raise RuntimeError(
            f"Dataset YAML has no usable '{split_name}' entry."
        )

    yaml_dir = data_yaml.parent
    base = yaml_dir

    if config.get("path"):
        candidate = Path(str(config["path"]))

        base = (
            candidate
            if candidate.is_absolute()
            else (yaml_dir / candidate).resolve()
        )

    split_path = Path(str(config[split_name]))

    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    if split_path.is_dir():
        images = sorted(
            [
                path
                for path in split_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in image_exts
            ],
            key=lambda item: str(item).lower(),
        )
    else:
        raise RuntimeError(
            f"Expected split directory but found: {split_path}"
        )

    instance_count = 0

    for image_path in images:
        text = str(image_path)

        if "\\images\\" in text:
            label_path = Path(
                text.replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
        else:
            label_path = Path(
                text.replace("/images/", "/labels/")
            ).with_suffix(".txt")

        if not label_path.is_file():
            continue

        instance_count += len(
            [
                line
                for line in label_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines()
                if line.strip()
            ]
        )

    return {
        "images": len(images),
        "instances": instance_count,
        "split_path": str(split_path),
    }


def load_stageb_train_args(
    args_yaml: Path,
    data_yaml: Path,
    output_root: Path,
):
    source = yaml.safe_load(
        args_yaml.read_text(encoding="utf-8")
    )

    if not isinstance(source, dict):
        raise RuntimeError(
            f"Invalid StageB args.yaml: {args_yaml}"
        )

    required_values = {
        "imgsz": 960,
        "batch": 6,
        "optimizer": "AdamW",
        "seed": 0,
        "deterministic": True,
    }

    discrepancies = []

    for key, expected in required_values.items():
        actual = source.get(key)

        if actual != expected:
            discrepancies.append(
                f"{key}: expected {expected!r}, found {actual!r}"
            )

    if discrepancies:
        raise RuntimeError(
            "StageB source protocol does not match the expected paper "
            "mainline configuration:\n  - "
            + "\n  - ".join(discrepancies)
        )

    blocked = {
        "task",
        "mode",
        "model",
        "data",
        "project",
        "name",
        "save_dir",
        "resume",
        "exist_ok",
        "pretrained",
    }

    train_args = {
        key: value
        for key, value in source.items()
        if key not in blocked
    }

    train_args.update(
        {
            "data": str(data_yaml),
            "project": str(
                output_root / "training_runs"
            ),
            "epochs": int(source.get("epochs", 120)),
            "imgsz": 960,
            "batch": 6,
            "optimizer": "AdamW",
            "seed": 0,
            "deterministic": True,
            "workers": 0,
            "device": 0,
            "resume": False,
            "pretrained": False,
            "save": True,
            "val": True,
            "exist_ok": False,
        }
    )

    return source, train_args


def transfer_stageb_weights(
    source_detector,
    target_detector,
):
    source_state = source_detector.model.state_dict()
    target_state = target_detector.model.state_dict()

    matched = {}
    matched_numel = 0
    target_numel = sum(
        tensor.numel()
        for tensor in target_state.values()
    )

    missing_or_incompatible = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)

        if (
            source_tensor is not None
            and tuple(source_tensor.shape)
            == tuple(target_tensor.shape)
        ):
            matched[key] = source_tensor
            matched_numel += target_tensor.numel()
        else:
            missing_or_incompatible.append(key)

    merged = dict(target_state)
    merged.update(matched)

    target_detector.model.load_state_dict(
        merged,
        strict=True,
    )

    coverage = (
        matched_numel / target_numel
        if target_numel > 0
        else 0.0
    )

    if coverage < 0.999:
        raise RuntimeError(
            "Shared-weight transfer coverage is below 99.9%. "
            f"Coverage={coverage:.6f}; unmatched tensors="
            f"{len(missing_or_incompatible)}. "
            "This would invalidate the paired continuation control."
        )

    return {
        "matched_tensor_count": len(matched),
        "target_tensor_count": len(target_state),
        "matched_parameter_coverage": coverage,
        "unmatched_tensor_count": len(
            missing_or_incompatible
        ),
    }


def calculate_structure(detector):
    from ultralytics.utils.torch_utils import get_flops

    detector.model.float().eval()

    params_m = sum(
        parameter.numel()
        for parameter in detector.model.parameters()
    ) / 1e6

    try:
        flops_g = float(
            get_flops(
                detector.model,
                imgsz=960,
            )
        )
    except Exception:
        flops_g = float("nan")

    return {
        "params_m": params_m,
        "flops_g": flops_g,
    }


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    data_yaml: Path,
    split_name: str,
    output_root: Path,
):
    from ultralytics import YOLO

    log("")
    log("=" * 96)
    log(f"[EVALUATE] {label} | split={split_name}")
    log(f"[CHECKPOINT] {checkpoint}")
    log(
        "[PROTOCOL] "
        f"imgsz={TEST_IMGSZ} | batch={TEST_BATCH} | "
        f"conf={TEST_CONF} | iou={TEST_IOU} | "
        f"max_det={TEST_MAX_DET} | augment=False"
    )
    log("=" * 96)

    detector = YOLO(str(checkpoint))

    patched = patch_runtime_attributes(detector.model)

    if patched:
        log(
            "[RUNTIME PATCH] expected_c1 restored for: "
            + ", ".join(patched)
        )

    metrics = detector.val(
        data=str(data_yaml),
        split=split_name,
        imgsz=TEST_IMGSZ,
        batch=TEST_BATCH,
        conf=TEST_CONF,
        iou=TEST_IOU,
        max_det=TEST_MAX_DET,
        augment=False,
        rect=False,
        half=False,
        device=TEST_DEVICE,
        workers=TEST_WORKERS,
        plots=False,
        save_json=False,
        project=str(output_root / "evaluations"),
        name=f"{label}_{split_name}",
        exist_ok=False,
        verbose=True,
    )

    structure = calculate_structure(detector)

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "ap50": float(metrics.box.map50),
        "ap50_95": float(metrics.box.map),
        "speed_inference_ms": float(
            metrics.speed.get("inference", float("nan"))
        ),
        **structure,
    }

    log(
        f"[RESULT] {label} | split={split_name} | "
        f"P={result['precision']:.6f} | "
        f"R={result['recall']:.6f} | "
        f"AP50={result['ap50']:.6f} | "
        f"AP50:95={result['ap50_95']:.6f}"
    )

    del detector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def train_variant(
    variant_name: str,
    expected_pool_class: str,
    model_yaml: Path,
    source_weight: Path,
    data_yaml: Path,
    train_args: dict,
    output_root: Path,
):
    from ultralytics import YOLO

    log("")
    log("=" * 96)
    log(f"[TRAIN] {variant_name}")
    log(f"[MODEL YAML] {model_yaml}")
    log(f"[INITIALIZATION] {source_weight}")
    log(f"[EXPECTED POOLING] {expected_pool_class}")
    log("=" * 96)

    source_detector = YOLO(str(source_weight))

    source_patched = patch_runtime_attributes(
        source_detector.model
    )

    if source_patched:
        log(
            "[SOURCE RUNTIME PATCH] "
            + ", ".join(source_patched)
        )

    target_detector = YOLO(str(model_yaml))

    target_patched = patch_runtime_attributes(
        target_detector.model
    )

    if target_patched:
        log(
            "[TARGET RUNTIME PATCH] "
            + ", ".join(target_patched)
        )

    source_pool = inspect_pool_module(
        source_detector,
        "SF1SPPF",
    )

    target_pool = inspect_pool_module(
        target_detector,
        expected_pool_class,
    )

    transfer = transfer_stageb_weights(
        source_detector,
        target_detector,
    )

    log(
        "[TRANSFER] "
        f"matched tensors={transfer['matched_tensor_count']}/"
        f"{transfer['target_tensor_count']} | "
        f"coverage={transfer['matched_parameter_coverage'] * 100:.4f}%"
    )

    del source_detector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_args = dict(train_args)

    current_args["name"] = variant_name

    log(
        f"[TRAIN START] {variant_name}. "
        "Epoch and validation progress will now be displayed by Ultralytics."
    )

    target_detector.train(**current_args)

    run_dir = (
        output_root
        / "training_runs"
        / variant_name
    )

    best_checkpoint = (
        run_dir
        / "weights"
        / "best.pt"
    )

    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training completed but best.pt was not found: "
            f"{best_checkpoint}"
        )

    log(
        f"[TRAIN COMPLETE] {variant_name} | "
        f"best checkpoint={best_checkpoint}"
    )

    return {
        "variant_name": variant_name,
        "expected_pool_class": expected_pool_class,
        "source_pool_module": source_pool,
        "target_pool_module": target_pool,
        "transfer": transfer,
        "best_checkpoint": str(best_checkpoint),
    }


def write_csv(path: Path, rows, fields):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--stageb-weight", required=True)
    parser.add_argument("--stageb-args", required=True)
    parser.add_argument("--soft-yaml", required=True)
    parser.add_argument("--max-yaml", required=True)
    parser.add_argument("--maxpool-source", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_yaml = Path(args.data).resolve()
    stageb_weight = Path(args.stageb_weight).resolve()
    stageb_args = Path(args.stageb_args).resolve()
    soft_yaml = Path(args.soft_yaml).resolve()
    max_yaml = Path(args.max_yaml).resolve()
    maxpool_source = Path(args.maxpool_source).resolve()
    output_root = Path(args.output).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    log("[PYTHON PHASE 1/6] Verifying inputs...")

    for title, path in (
        ("Project root", root),
        ("Data YAML", data_yaml),
        ("StageB source checkpoint", stageb_weight),
        ("StageB source args.yaml", stageb_args),
        ("SoftPool YAML", soft_yaml),
        ("MaxPool YAML", max_yaml),
        ("MaxPool source", maxpool_source),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{title} does not exist: {path}"
            )

        log(f"    [OK] {title}: {path}")

    log("[PYTHON PHASE 1/6] Completed.")

    log("")
    log("[PYTHON PHASE 2/6] Auditing dataset splits...")

    val_info = count_split(data_yaml, "val")
    test_info = count_split(data_yaml, "test")

    log(
        f"    Validation: {val_info['images']} images | "
        f"{val_info['instances']} instances"
    )

    log(
        f"    Test: {test_info['images']} images | "
        f"{test_info['instances']} instances"
    )

    if test_info["images"] != 572:
        raise RuntimeError(
            "Fixed WTMD test split mismatch. "
            f"Expected 572 images, found "
            f"{test_info['images']}."
        )

    if test_info["instances"] != 2301:
        raise RuntimeError(
            "Fixed WTMD test instance mismatch. "
            f"Expected 2301 instances, found "
            f"{test_info['instances']}."
        )

    log("[PYTHON PHASE 2/6] Completed.")

    log("")
    log("[PYTHON PHASE 3/6] Verifying architectures and registering modules...")

    verify_yaml_pair(
        soft_yaml,
        max_yaml,
    )

    registered_modules = register_custom_modules(
        root,
        maxpool_source,
    )

    stageb_sha = sha256_with_progress(
        stageb_weight,
        "StageB SoftPool source checkpoint",
    )

    source_args, train_args = load_stageb_train_args(
        stageb_args,
        data_yaml,
        output_root,
    )

    (output_root / "source_stageb_training_args.json").write_text(
        json.dumps(
            source_args,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (output_root / "paired_continuation_training_args.json").write_text(
        json.dumps(
            train_args,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("[PYTHON PHASE 3/6] Completed.")

    log("")
    log("[PYTHON PHASE 4/6] Evaluating current StageB SoftPool reference checkpoint...")

    reference_val = evaluate_checkpoint(
        label="StageB_SoftPool_Reference",
        checkpoint=stageb_weight,
        data_yaml=data_yaml,
        split_name="val",
        output_root=output_root,
    )

    reference_test = evaluate_checkpoint(
        label="StageB_SoftPool_Reference",
        checkpoint=stageb_weight,
        data_yaml=data_yaml,
        split_name="test",
        output_root=output_root,
    )

    log("[PYTHON PHASE 4/6] Completed.")

    log("")
    log("[PYTHON PHASE 5/6] Training paired continuation controls...")

    soft_run = train_variant(
        variant_name="StageB_Initialized_SoftPool_Continuation",
        expected_pool_class="SF1SPPF",
        model_yaml=soft_yaml,
        source_weight=stageb_weight,
        data_yaml=data_yaml,
        train_args=train_args,
        output_root=output_root,
    )

    max_run = train_variant(
        variant_name="StageB_Initialized_MaxPool_Continuation",
        expected_pool_class="MaxPoolSF1SPPF",
        model_yaml=max_yaml,
        source_weight=stageb_weight,
        data_yaml=data_yaml,
        train_args=train_args,
        output_root=output_root,
    )

    log("[PYTHON PHASE 5/6] Completed.")

    log("")
    log("[PYTHON PHASE 6/6] Running final validation and held-out test evaluation...")

    soft_best = Path(
        soft_run["best_checkpoint"]
    )

    max_best = Path(
        max_run["best_checkpoint"]
    )

    soft_val = evaluate_checkpoint(
        label="StageB_Initialized_SoftPool_Continuation",
        checkpoint=soft_best,
        data_yaml=data_yaml,
        split_name="val",
        output_root=output_root,
    )

    soft_test = evaluate_checkpoint(
        label="StageB_Initialized_SoftPool_Continuation",
        checkpoint=soft_best,
        data_yaml=data_yaml,
        split_name="test",
        output_root=output_root,
    )

    max_val = evaluate_checkpoint(
        label="StageB_Initialized_MaxPool_Continuation",
        checkpoint=max_best,
        data_yaml=data_yaml,
        split_name="val",
        output_root=output_root,
    )

    max_test = evaluate_checkpoint(
        label="StageB_Initialized_MaxPool_Continuation",
        checkpoint=max_best,
        data_yaml=data_yaml,
        split_name="test",
        output_root=output_root,
    )

    rows = [
        {
            "model": "StageB SoftPool reference",
            "pooling_operator": "SoftPool-SPPF",
            "role": "Main fixed checkpoint",
            "initialization": "Not applicable",
            "checkpoint": str(stageb_weight),
            "checkpoint_sha256": stageb_sha,
            "val_precision": reference_val["precision"],
            "val_recall": reference_val["recall"],
            "val_ap50": reference_val["ap50"],
            "val_ap50_95": reference_val["ap50_95"],
            "test_precision": reference_test["precision"],
            "test_recall": reference_test["recall"],
            "test_ap50": reference_test["ap50"],
            "test_ap50_95": reference_test["ap50_95"],
            "params_m": reference_test["params_m"],
            "flops_g": reference_test["flops_g"],
            "test_inference_ms": reference_test["speed_inference_ms"],
            "weight_transfer_coverage": "",
        },
        {
            "model": "StageB-initialized SoftPool continuation",
            "pooling_operator": "SoftPool-SPPF",
            "role": "Paired continuation control",
            "initialization": str(stageb_weight),
            "checkpoint": str(soft_best),
            "checkpoint_sha256": sha256_with_progress(
                soft_best,
                "SoftPool continuation best checkpoint",
            ),
            "val_precision": soft_val["precision"],
            "val_recall": soft_val["recall"],
            "val_ap50": soft_val["ap50"],
            "val_ap50_95": soft_val["ap50_95"],
            "test_precision": soft_test["precision"],
            "test_recall": soft_test["recall"],
            "test_ap50": soft_test["ap50"],
            "test_ap50_95": soft_test["ap50_95"],
            "params_m": soft_test["params_m"],
            "flops_g": soft_test["flops_g"],
            "test_inference_ms": soft_test["speed_inference_ms"],
            "weight_transfer_coverage": (
                soft_run["transfer"][
                    "matched_parameter_coverage"
                ]
            ),
        },
        {
            "model": "StageB-initialized MaxPool continuation",
            "pooling_operator": "MaxPool-SPPF",
            "role": "Paired continuation control",
            "initialization": str(stageb_weight),
            "checkpoint": str(max_best),
            "checkpoint_sha256": sha256_with_progress(
                max_best,
                "MaxPool continuation best checkpoint",
            ),
            "val_precision": max_val["precision"],
            "val_recall": max_val["recall"],
            "val_ap50": max_val["ap50"],
            "val_ap50_95": max_val["ap50_95"],
            "test_precision": max_test["precision"],
            "test_recall": max_test["recall"],
            "test_ap50": max_test["ap50"],
            "test_ap50_95": max_test["ap50_95"],
            "params_m": max_test["params_m"],
            "flops_g": max_test["flops_g"],
            "test_inference_ms": max_test["speed_inference_ms"],
            "weight_transfer_coverage": (
                max_run["transfer"][
                    "matched_parameter_coverage"
                ]
            ),
        },
    ]

    summary_path = (
        output_root
        / "stageb_initialized_softpool_maxpool_summary.csv"
    )

    fields = [
        "model",
        "pooling_operator",
        "role",
        "initialization",
        "checkpoint",
        "checkpoint_sha256",
        "val_precision",
        "val_recall",
        "val_ap50",
        "val_ap50_95",
        "test_precision",
        "test_recall",
        "test_ap50",
        "test_ap50_95",
        "params_m",
        "flops_g",
        "test_inference_ms",
        "weight_transfer_coverage",
    ]

    write_csv(
        summary_path,
        rows,
        fields,
    )

    manifest = {
        "purpose": (
            "StageB-initialized paired SoftPool-SPPF versus "
            "MaxPool-SPPF continuation control"
        ),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_checkpoint": str(stageb_weight),
        "source_checkpoint_sha256": stageb_sha,
        "data_yaml": str(data_yaml),
        "validation_split": val_info,
        "test_split": test_info,
        "test_protocol": {
            "split": "test",
            "imgsz": TEST_IMGSZ,
            "batch": TEST_BATCH,
            "conf": TEST_CONF,
            "iou": TEST_IOU,
            "max_det": TEST_MAX_DET,
            "augment": False,
            "rect": False,
            "half": False,
            "device": TEST_DEVICE,
            "workers": TEST_WORKERS,
        },
        "training_args": train_args,
        "registered_modules": registered_modules,
        "soft_run": soft_run,
        "max_run": max_run,
        "summary_csv": str(summary_path),
    }

    manifest_path = (
        output_root
        / "stageb_initialized_softpool_maxpool_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("")
    log("=" * 96)
    log("STAGEB_INITIALIZED_SOFTPOOL_MAXPOOL_CONTROL_COMPLETED")
    log(f"SUMMARY_CSV={summary_path}")
    log(f"MANIFEST_JSON={manifest_path}")
    log(
        "Main reference test AP50="
        f"{reference_test['ap50']:.6f}"
    )
    log(
        "SoftPool continuation test AP50="
        f"{soft_test['ap50']:.6f}"
    )
    log(
        "MaxPool continuation test AP50="
        f"{max_test['ap50']:.6f}"
    )
    log("=" * 96)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback.print_exc()
        raise
