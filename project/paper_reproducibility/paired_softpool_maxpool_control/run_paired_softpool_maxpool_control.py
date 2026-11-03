import copy
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ["PAIR_ROOT"])
SOURCE_PT = Path(os.environ["PAIR_SOURCE_PT"])
DATA_YAML = Path(os.environ["PAIR_DATA_YAML"])
ARTIFACT_ROOT = Path(os.environ["PAIR_ARTIFACT_ROOT"])
RUN_TAG = os.environ["PAIR_RUN_TAG"]

SEED = int(os.environ["PAIR_SEED"])
IMGSZ = int(os.environ["PAIR_IMGSZ"])
BATCH = int(os.environ["PAIR_BATCH"])
EPOCHS = int(os.environ["PAIR_EPOCHS"])
WORKERS = int(os.environ["PAIR_WORKERS"])
AMP = os.environ["PAIR_AMP"].strip().lower() == "true"
OPTIMIZER = os.environ["PAIR_OPTIMIZER"]
LR0 = float(os.environ["PAIR_LR0"])
PATIENCE = int(os.environ["PAIR_PATIENCE"])
CLOSE_MOSAIC = int(os.environ["PAIR_CLOSE_MOSAIC"])

SITE_PACKAGES = ROOT / "venv_8363" / "Lib" / "site-packages"

if not SOURCE_PT.is_file():
    raise FileNotFoundError(f"Source checkpoint not found: {SOURCE_PT}")

if not DATA_YAML.is_file():
    raise FileNotFoundError(f"Data YAML not found: {DATA_YAML}")

if not SITE_PACKAGES.is_dir():
    raise FileNotFoundError(f"site-packages not found: {SITE_PACKAGES}")

ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

for path in (str(ARTIFACT_ROOT), str(SITE_PACKAGES)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.nn as nn
import yaml
import ultralytics
from ultralytics import YOLO

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def to_plain(value):
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, type):
        return value.__name__

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)

def class_path(obj):
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

def materialize_model(yolo_model):
    net = yolo_model.model.to("cpu").eval()

    with torch.no_grad():
        _ = net(torch.zeros(1, 3, 128, 128))

    net.train()

def model_signature(yolo_model):
    net = yolo_model.model
    total_params = sum(p.numel() for p in net.parameters())
    strides = []

    if hasattr(net, "stride"):
        try:
            strides = [
                round(float(v), 3)
                for v in net.stride.detach().cpu().flatten().tolist()
            ]
        except Exception:
            strides = [str(v) for v in net.stride]

    pooling_modules = []

    for name, module in net.named_modules():
        module_name = module.__class__.__name__

        if module_name in {"SF1SPPF", "MaxPoolSF1SPPF"}:
            pool = getattr(module, "pool", None)

            pooling_modules.append({
                "name": name,
                "module_class": class_path(module),
                "pool_class": class_path(pool) if pool is not None else None,
                "fixed_c1": to_plain(getattr(module, "fixed_c1", None)),
                "c2": to_plain(getattr(module, "c2", None)),
                "k": to_plain(getattr(module, "k", None)),
            })

    return {
        "params": int(total_params),
        "params_m": round(total_params / 1e6, 6),
        "strides": strides,
        "pooling_modules": pooling_modules,
    }

def metrics_dict(metrics):
    out = {}

    for key, value in metrics.results_dict.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            out[str(key)] = str(value)

    return out

def find_metric(data, exact_keys):
    for key in exact_keys:
        if key in data:
            return data[key]

    for key, value in data.items():
        lower_key = key.lower()

        if any(token.lower() in lower_key for token in exact_keys):
            return value

    return None

def replace_sf1sppf_with_maxpool(value):
    replacement_count = {"n": 0}

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}

        if isinstance(x, list):
            return [walk(v) for v in x]

        if isinstance(x, str):
            short_name = x.split(".")[-1]

            if short_name == "SF1SPPF":
                replacement_count["n"] += 1
                return "MaxPoolSF1SPPF"

            return x

        if isinstance(x, type) and x.__name__ == "SF1SPPF":
            replacement_count["n"] += 1
            return "MaxPoolSF1SPPF"

        return x

    return walk(value), replacement_count["n"]

def standard_validation(yolo_model, split, project_dir, run_name):
    result = yolo_model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=0.001,
        iou=0.50,
        max_det=10000,
        augment=False,
        half=False,
        device=0,
        workers=WORKERS,
        plots=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )

    return metrics_dict(result)

# ------------------------------------------------------------
# Reuse only an already completed and fully traceable paired run.
# Historical loose files are not accepted as direct evidence.
# ------------------------------------------------------------

SOURCE_HASH = sha256(SOURCE_PT)
DATA_HASH = sha256(DATA_YAML)

for existing_summary in ARTIFACT_ROOT.rglob("paired_softpool_maxpool_summary.json"):
    try:
        old = json.loads(existing_summary.read_text(encoding="utf-8"))

        valid = (
            old.get("status") == "completed"
            and old.get("source_checkpoint_sha256") == SOURCE_HASH
            and old.get("data_yaml_sha256") == DATA_HASH
            and old.get("pair_protocol", {}).get("seed") == SEED
            and old.get("pair_protocol", {}).get("imgsz") == IMGSZ
            and old.get("pair_protocol", {}).get("batch") == BATCH
            and old.get("pair_protocol", {}).get("epochs") == EPOCHS
            and old.get("pair_protocol", {}).get("optimizer") == OPTIMIZER
            and old.get("pair_protocol", {}).get("lr0") == LR0
            and old.get("pair_protocol", {}).get("close_mosaic") == CLOSE_MOSAIC
        )

        if valid:
            print("=" * 88)
            print("EXISTING VERIFIED PAIRED EXPERIMENT FOUND")
            print("=" * 88)
            print(f"SUMMARY_JSON={existing_summary}")
            print(
                "SOFTPOOL_AP50=",
                old["paired_softpool_validation"].get("metrics/mAP50(B)")
            )
            print(
                "SOFTPOOL_AP50_95=",
                old["paired_softpool_validation"].get("metrics/mAP50-95(B)")
            )
            print(
                "MAXPOOL_AP50=",
                old["paired_maxpool_validation"].get("metrics/mAP50(B)")
            )
            print(
                "MAXPOOL_AP50_95=",
                old["paired_maxpool_validation"].get("metrics/mAP50-95(B)")
            )
            print("No new training was started.")
            print("=" * 88)
            raise SystemExit(0)

    except SystemExit:
        raise
    except Exception:
        pass

# ------------------------------------------------------------
# Start a new traceable paired experiment.
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Stop to prevent an unintended CPU experiment."
    )

DEVICE_NAME = torch.cuda.get_device_name(0)

RUN_DIR = ARTIFACT_ROOT / RUN_TAG
RUN_DIR.mkdir(parents=True, exist_ok=False)

PAIR_RUNS = RUN_DIR / "runs"
PAIR_RUNS.mkdir(parents=True, exist_ok=True)

SOFT_YAML = RUN_DIR / "stageB_softpool_extracted.yaml"
MAX_YAML = RUN_DIR / "stageB_maxpool_control.yaml"
INIT_STATE = RUN_DIR / "shared_random_initialization.pt"
CUSTOM_MODULE = ARTIFACT_ROOT / "maxpool_sf1sppf_control.py"
SUMMARY_JSON = RUN_DIR / "paired_softpool_maxpool_summary.json"

# ------------------------------------------------------------
# Make original checkpoint class aliases resolvable.
# ------------------------------------------------------------

import ultralytics.nn.modules.sf1_sppf as sf1_sppf_module
sys.modules["sf1_sppf"] = sf1_sppf_module

from ultralytics.nn.modules.sf1_sppf import register_ultralytics_modules
register_ultralytics_modules()

# ------------------------------------------------------------
# Create persistent strict MaxPool control module.
# ------------------------------------------------------------

CUSTOM_MODULE.write_text(
r'''
import torch.nn as nn

from ultralytics.nn.modules.sf1_sppf import SF1SPPF


class MaxPoolSF1SPPF(SF1SPPF):
    """
    Strict matched control for SF1SPPF.

    Preserved:
      - original argument parsing;
      - original delayed layer construction;
      - optional channel adapter;
      - cv1 and cv2;
      - hidden-channel logic;
      - three serial pooling operations;
      - feature-concatenation order.

    Changed:
      SoftPool2d -> nn.MaxPool2d only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(
            kernel_size=self.k,
            stride=1,
            padding=self.k // 2,
        )


def register_maxpool_sf1sppf():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["MaxPoolSF1SPPF"] = MaxPoolSF1SPPF
    modules.__dict__["MaxPoolSF1SPPF"] = MaxPoolSF1SPPF

    setattr(tasks, "MaxPoolSF1SPPF", MaxPoolSF1SPPF)
    setattr(modules, "MaxPoolSF1SPPF", MaxPoolSF1SPPF)

    if hasattr(modules, "__all__") and isinstance(modules.__all__, list):
        if "MaxPoolSF1SPPF" not in modules.__all__:
            modules.__all__.append("MaxPoolSF1SPPF")
''',
encoding="utf-8",
)

custom_control = importlib.import_module("maxpool_sf1sppf_control")
custom_control.register_maxpool_sf1sppf()
MaxPoolSF1SPPF = custom_control.MaxPoolSF1SPPF

# ------------------------------------------------------------
# Load and verify the actual final SoftPool StageB checkpoint.
# ------------------------------------------------------------

source_model = YOLO(str(SOURCE_PT))
source_signature = model_signature(source_model)

if abs(source_signature["params_m"] - 1.467699) > 0.01:
    raise RuntimeError(
        f"Unexpected StageB parameter count: {source_signature['params_m']} M."
    )

if source_signature["strides"] != [4.0, 8.0, 16.0]:
    raise RuntimeError(
        f"Unexpected StageB strides: {source_signature['strides']}."
    )

if len(source_signature["pooling_modules"]) != 1:
    raise RuntimeError(
        "Expected exactly one SF1SPPF module in the final StageB checkpoint."
    )

source_pool_class = source_signature["pooling_modules"][0]["pool_class"]

if source_pool_class is None or "SoftPool2d" not in source_pool_class:
    raise RuntimeError(
        f"Final StageB checkpoint is not confirmed SoftPool2d. Found: {source_pool_class}"
    )

# ------------------------------------------------------------
# Re-validate the actual final StageB checkpoint first.
# These are the values used to verify manuscript Table results.
# ------------------------------------------------------------

print("=" * 88)
print("FINAL STAGEB SOURCE CHECKPOINT VERIFIED")
print("=" * 88)
print(f"SOURCE_PT={SOURCE_PT}")
print(f"SOURCE_PT_SHA256={SOURCE_HASH}")
print(f"SOURCE_PARAMS_M={source_signature['params_m']}")
print(f"SOURCE_STRIDES={source_signature['strides']}")
print(f"SOURCE_POOL={source_pool_class}")
print(f"DATA_YAML={DATA_YAML}")
print(f"DATA_YAML_SHA256={DATA_HASH}")
print(f"GPU={DEVICE_NAME}")
print("=" * 88)

source_val_metrics = standard_validation(
    source_model,
    split="val",
    project_dir=PAIR_RUNS,
    run_name="FinalStageB_SoftPool_Val",
)

source_test_metrics = standard_validation(
    source_model,
    split="test",
    project_dir=PAIR_RUNS,
    run_name="FinalStageB_SoftPool_Test",
)

print("SOURCE_FINAL_STAGEB_VALIDATION")
print(
    "SOURCE_VAL_AP50=",
    find_metric(source_val_metrics, ["metrics/mAP50(B)", "mAP50"]),
)
print(
    "SOURCE_VAL_AP50_95=",
    find_metric(source_val_metrics, ["metrics/mAP50-95(B)", "mAP50-95"]),
)
print(
    "SOURCE_TEST_AP50=",
    find_metric(source_test_metrics, ["metrics/mAP50(B)", "mAP50"]),
)
print(
    "SOURCE_TEST_AP50_95=",
    find_metric(source_test_metrics, ["metrics/mAP50-95(B)", "mAP50-95"]),
)

# ------------------------------------------------------------
# Export exact source architecture from the verified checkpoint.
# ------------------------------------------------------------

source_yaml = to_plain(copy.deepcopy(source_model.model.yaml))
source_yaml.pop("yaml_file", None)

with SOFT_YAML.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        source_yaml,
        f,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

max_yaml, replacement_count = replace_sf1sppf_with_maxpool(source_yaml)

if replacement_count != 1:
    raise RuntimeError(
        f"Expected exactly one SF1SPPF replacement, found {replacement_count}."
    )

with MAX_YAML.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        max_yaml,
        f,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

# ------------------------------------------------------------
# Create the paired SoftPool and MaxPool models.
# Both begin from exactly the same saved random initialization.
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

soft_pair_model = YOLO(str(SOFT_YAML))
max_pair_model = YOLO(str(MAX_YAML))

materialize_model(soft_pair_model)
materialize_model(max_pair_model)

shared_initial_state = {
    key: value.detach().cpu().clone()
    for key, value in soft_pair_model.model.state_dict().items()
}

torch.save(shared_initial_state, INIT_STATE)

max_pair_model.model.load_state_dict(shared_initial_state, strict=True)

soft_pair_signature = model_signature(soft_pair_model)
max_pair_signature = model_signature(max_pair_model)

if abs(soft_pair_signature["params_m"] - max_pair_signature["params_m"]) > 0.000001:
    raise RuntimeError(
        "SoftPool and MaxPool control parameter counts do not match."
    )

if soft_pair_signature["strides"] != max_pair_signature["strides"]:
    raise RuntimeError(
        "SoftPool and MaxPool control strides do not match."
    )

if soft_pair_signature["strides"] != [4.0, 8.0, 16.0]:
    raise RuntimeError(
        "Paired models do not retain the P2/P3/P4 prediction hierarchy."
    )

if len(soft_pair_signature["pooling_modules"]) != 1:
    raise RuntimeError("Paired SoftPool model has no unique SF1SPPF module.")

if len(max_pair_signature["pooling_modules"]) != 1:
    raise RuntimeError("Paired MaxPool model has no unique MaxPoolSF1SPPF module.")

if "SoftPool2d" not in soft_pair_signature["pooling_modules"][0]["pool_class"]:
    raise RuntimeError("Paired SoftPool branch did not retain SoftPool2d.")

if max_pair_signature["pooling_modules"][0]["pool_class"] != "torch.nn.modules.pooling.MaxPool2d":
    raise RuntimeError("Paired MaxPool branch did not use nn.MaxPool2d.")

print("=" * 88)
print("PAIRED ARCHITECTURE CHECK")
print("=" * 88)
print(f"PAIRED_SOFTPOOL_PARAMS_M={soft_pair_signature['params_m']}")
print(f"PAIRED_MAXPOOL_PARAMS_M={max_pair_signature['params_m']}")
print(f"PAIRED_STRIDES={soft_pair_signature['strides']}")
print(f"PAIRED_SOFTPOOL_POOL={soft_pair_signature['pooling_modules'][0]['pool_class']}")
print(f"PAIRED_MAXPOOL_POOL={max_pair_signature['pooling_modules'][0]['pool_class']}")
print(f"SHARED_INITIALIZATION={INIT_STATE}")
print(f"SHARED_INITIALIZATION_SHA256={sha256(INIT_STATE)}")
print("PAIRED_ARCHITECTURE_CHECK=PASS")
print("=" * 88)

# ------------------------------------------------------------
# Train SoftPool branch.
# ------------------------------------------------------------

soft_pair_model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    optimizer=OPTIMIZER,
    lr0=LR0,
    patience=PATIENCE,
    workers=WORKERS,
    seed=SEED,
    deterministic=True,
    pretrained=False,
    close_mosaic=CLOSE_MOSAIC,
    amp=AMP,
    device=0,
    project=str(PAIR_RUNS),
    name="Paired_SoftPool",
    exist_ok=False,
    save=True,
    plots=True,
    val=True,
    verbose=True,
)

soft_best_pt = Path(soft_pair_model.trainer.save_dir) / "weights" / "best.pt"

if not soft_best_pt.is_file():
    raise FileNotFoundError(f"SoftPool paired best.pt missing: {soft_best_pt}")

# ------------------------------------------------------------
# Train MaxPool branch from the identical saved initialization.
# ------------------------------------------------------------

max_pair_model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    optimizer=OPTIMIZER,
    lr0=LR0,
    patience=PATIENCE,
    workers=WORKERS,
    seed=SEED,
    deterministic=True,
    pretrained=False,
    close_mosaic=CLOSE_MOSAIC,
    amp=AMP,
    device=0,
    project=str(PAIR_RUNS),
    name="Paired_MaxPool",
    exist_ok=False,
    save=True,
    plots=True,
    val=True,
    verbose=True,
)

max_best_pt = Path(max_pair_model.trainer.save_dir) / "weights" / "best.pt"

if not max_best_pt.is_file():
    raise FileNotFoundError(f"MaxPool paired best.pt missing: {max_best_pt}")

# ------------------------------------------------------------
# Reload both best checkpoints and evaluate them using one
# identical validation protocol.
# ------------------------------------------------------------

best_soft = YOLO(str(soft_best_pt))
best_max = YOLO(str(max_best_pt))

soft_best_signature = model_signature(best_soft)
max_best_signature = model_signature(best_max)

if "SoftPool2d" not in soft_best_signature["pooling_modules"][0]["pool_class"]:
    raise RuntimeError("Saved paired SoftPool checkpoint lost SoftPool2d.")

if max_best_signature["pooling_modules"][0]["pool_class"] != "torch.nn.modules.pooling.MaxPool2d":
    raise RuntimeError("Saved paired MaxPool checkpoint lost MaxPool2d.")

paired_soft_metrics = standard_validation(
    best_soft,
    split="val",
    project_dir=PAIR_RUNS,
    run_name="Paired_SoftPool_StandardVal",
)

paired_max_metrics = standard_validation(
    best_max,
    split="val",
    project_dir=PAIR_RUNS,
    run_name="Paired_MaxPool_StandardVal",
)

soft_ap50 = find_metric(paired_soft_metrics, ["metrics/mAP50(B)", "mAP50"])
soft_ap5095 = find_metric(paired_soft_metrics, ["metrics/mAP50-95(B)", "mAP50-95"])

max_ap50 = find_metric(paired_max_metrics, ["metrics/mAP50(B)", "mAP50"])
max_ap5095 = find_metric(paired_max_metrics, ["metrics/mAP50-95(B)", "mAP50-95"])

delta_ap50 = None if soft_ap50 is None or max_ap50 is None else soft_ap50 - max_ap50
delta_ap5095 = (
    None
    if soft_ap5095 is None or max_ap5095 is None
    else soft_ap5095 - max_ap5095
)

summary = {
    "status": "completed",
    "experiment": RUN_TAG,
    "purpose": (
        "Paired SoftPool-SPPF versus MaxPool-SPPF control derived from the "
        "verified final StageB architecture. Both branches used identical "
        "random initialization and training settings. Only the pooling "
        "operator was changed."
    ),
    "environment": {
        "python": sys.executable,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "gpu": DEVICE_NAME,
        "cuda_available": torch.cuda.is_available(),
    },
    "source_checkpoint": str(SOURCE_PT),
    "source_checkpoint_sha256": SOURCE_HASH,
    "data_yaml": str(DATA_YAML),
    "data_yaml_sha256": DATA_HASH,
    "source_checkpoint_signature": source_signature,
    "source_final_stageB_validation": source_val_metrics,
    "source_final_stageB_test": source_test_metrics,
    "extracted_softpool_yaml": str(SOFT_YAML),
    "extracted_softpool_yaml_sha256": sha256(SOFT_YAML),
    "maxpool_control_yaml": str(MAX_YAML),
    "maxpool_control_yaml_sha256": sha256(MAX_YAML),
    "maxpool_control_module": str(CUSTOM_MODULE),
    "maxpool_control_module_sha256": sha256(CUSTOM_MODULE),
    "shared_initialization": str(INIT_STATE),
    "shared_initialization_sha256": sha256(INIT_STATE),
    "pair_protocol": {
        "seed": SEED,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "epochs": EPOCHS,
        "workers": WORKERS,
        "amp": AMP,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "patience": PATIENCE,
        "close_mosaic": CLOSE_MOSAIC,
        "pretrained": False,
        "validation_split": "val",
        "validation_conf": 0.001,
        "validation_iou": 0.50,
        "validation_max_det": 10000,
        "validation_augment": False,
    },
    "paired_softpool_pretraining_signature": soft_pair_signature,
    "paired_maxpool_pretraining_signature": max_pair_signature,
    "paired_softpool_best_checkpoint": str(soft_best_pt),
    "paired_softpool_best_checkpoint_sha256": sha256(soft_best_pt),
    "paired_softpool_reloaded_signature": soft_best_signature,
    "paired_maxpool_best_checkpoint": str(max_best_pt),
    "paired_maxpool_best_checkpoint_sha256": sha256(max_best_pt),
    "paired_maxpool_reloaded_signature": max_best_signature,
    "paired_softpool_validation": paired_soft_metrics,
    "paired_maxpool_validation": paired_max_metrics,
    "comparison": {
        "softpool_ap50": soft_ap50,
        "softpool_ap50_95": soft_ap5095,
        "maxpool_ap50": max_ap50,
        "maxpool_ap50_95": max_ap5095,
        "softpool_minus_maxpool_ap50": delta_ap50,
        "softpool_minus_maxpool_ap50_95": delta_ap5095,
    },
}

SUMMARY_JSON.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("=" * 88)
print("PAIRED SOFTPOOL--MAXPOOL EXPERIMENT COMPLETED")
print("=" * 88)
print(f"SUMMARY_JSON={SUMMARY_JSON}")
print(f"SOFTPOOL_AP50={soft_ap50}")
print(f"SOFTPOOL_AP50_95={soft_ap5095}")
print(f"MAXPOOL_AP50={max_ap50}")
print(f"MAXPOOL_AP50_95={max_ap5095}")
print(f"SOFTPOOL_MINUS_MAXPOOL_AP50={delta_ap50}")
print(f"SOFTPOOL_MINUS_MAXPOOL_AP50_95={delta_ap5095}")
print("=" * 88)
