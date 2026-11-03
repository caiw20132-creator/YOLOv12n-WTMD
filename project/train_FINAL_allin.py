# -*- coding: utf-8 -*-
"""
================================================================================
FINAL TRAINING SCRIPT (NO CLI ARGS NEEDED)
File: D:\YOLO-WTB\train_FINAL_allin.py

功能：
- 自动生成 all-in holdout 数据 yaml（若不存在）
- 自动注册 SF1SPPF
- 自动按 StageA(896) -> StageB(960) 两阶段训练
- 自动评估：holdout(val) + (可选) 原始 test split

你只需要运行：
  python D:\YOLO-WTB\train_FINAL_allin.py

================================================================================
"""

import os
import sys
import time
import random
import subprocess
from pathlib import Path
from typing import Iterable, Set, Tuple, Optional

# -------------------------
# 0) 固定你的工程路径（按你现在的真实路径写死）
# -------------------------
ROOT = Path(r"D:\YOLO-WTB")

MODEL_YAML = ROOT / "yolo12n_damage_p4lite_enhanced_p234_v2.yaml"
SF1_PY     = ROOT / "sf1_sppf.py"

# 你这次 all-in 的源数据（包含 train/valid/test 的那个）
SRC_YAML   = ROOT / "data_fix_all_v1.yaml"

# all-in holdout 输出
HOLDOUT_YAML = ROOT / "data_allin_holdout.yaml"
HOLDOUT_DIR  = ROOT / "allin_lists"
MAKE_HOLDOUT_TOOL = ROOT / "tools" / "make_allin_holdout_yaml.py"

# 预训练初始化权重（你现在最强的 T2 best）
INIT_WEIGHTS = ROOT / r"runs_wtb\T2_full832_ft_from_tile512_best_lr5e-5\weights\best.pt"

# 输出 runs 根目录
RUNS_DIR = ROOT / "runs_wtb"

# -------------------------
# 1) 训练超参（最终版）
# -------------------------
SEED = 0
DEVICE = "0"
WORKERS = 4
AMP = True

# holdout 比例：训练会用“全部图像的 92%”，留 8% 做最终验证（强烈建议保留）
HOLDOUT_RATIO = 0.08

STAGE_A = dict(
    name="FINAL_allin_896_stageA",
    imgsz=896,
    batch=4,
    epochs=220,
    patience=50,
    optimizer="AdamW",
    lr0=2e-4,
    lrf=0.02,
    cos_lr=True,
    weight_decay=0.01,
    mosaic=0.10,
    close_mosaic=15,
    mixup=0.0,
    copy_paste=0.05,
)

STAGE_B = dict(
    name="FINAL_allin_960_stageB",
    imgsz=960,
    batch=3,
    epochs=80,
    patience=30,
    optimizer="AdamW",
    lr0=5e-5,
    lrf=0.01,
    cos_lr=True,
    weight_decay=0.01,
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    copy_paste=0.0,
)

# 是否额外在“原始 data_fix_all_v1.yaml 的 test split”上跑一次评估
EVAL_ON_SRC_TEST = True


# =============================================================================
# Utils
# =============================================================================
def _log(msg: str) -> None:
    print(msg, flush=True)


def set_reproducible(seed: int = 0) -> None:
    # 尽量可复现（Windows + CUDA 下也不保证 100%）
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def ensure_allin_holdout_yaml() -> None:
    """
    若 HOLDOUT_YAML 不存在，则调用 tools\make_allin_holdout_yaml.py 自动生成：
      total_images : 5744
      train_all    : 5284
      val_holdout  : 460
    """
    if HOLDOUT_YAML.exists() and HOLDOUT_DIR.exists():
        _log(f"[OK] all-in holdout yaml exists: {HOLDOUT_YAML}")
        return

    if not MAKE_HOLDOUT_TOOL.exists():
        raise FileNotFoundError(f"Missing tool: {MAKE_HOLDOUT_TOOL}")

    _log("[INFO] all-in holdout yaml not found -> generating ...")
    cmd = [
        sys.executable, str(MAKE_HOLDOUT_TOOL),
        "--src_yaml", str(SRC_YAML),
        "--out_yaml", str(HOLDOUT_YAML),
        "--out_dir",  str(HOLDOUT_DIR),
        "--holdout_ratio", str(HOLDOUT_RATIO),
        "--seed", str(SEED),
    ]
    _log("[RUN] " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"make_allin_holdout_yaml failed with code {r.returncode}")

    if not HOLDOUT_YAML.exists():
        raise RuntimeError("Generated holdout yaml missing (unexpected).")

    _log(f"[OK] generated: {HOLDOUT_YAML}")


def register_sf1_modules() -> None:
    """
    载入 sf1_sppf.py 并注册到 ultralytics 解析器（与你 train_wtb_sf1_final.py 的行为一致）
    """
    if not SF1_PY.exists():
        raise FileNotFoundError(f"Missing SF1 file: {SF1_PY}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("sf1_sppf", str(SF1_PY))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for sf1_sppf.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if hasattr(mod, "register_ultralytics_modules"):
        _log(f"[SF1] loading: {SF1_PY}")
        mod.register_ultralytics_modules()  # type: ignore
        _log("[OK] SF1SPPF registered into ultralytics.nn.tasks")
    else:
        _log("[WARN] sf1_sppf.py has no register_ultralytics_modules() - skip registration")


def _img_to_label_path(img_path: Path) -> Path:

    parts = list(img_path.parts)
    # 找到 images 目录并替换为 labels
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        i = lowered.index("images")
        parts[i] = "labels"
        lp = Path(*parts)
        return lp.with_suffix(".txt")

    # fallback：同级 labels
    lp = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    return lp


def check_single_class_from_lists(list_files: Iterable[Path], max_images: int = 5000) -> Tuple[bool, Set[int]]:
    """
    扫描 list_files（txt 内每行是 image path），抽检/全检 labels 里的 class id。
    返回：是否全为 {0}，以及出现过的 class_id 集合。
    """
    class_ids: Set[int] = set()
    scanned = 0
    missing_label = 0

    for lf in list_files:
        if not lf.exists():
            _log(f"[WARN] list file not found: {lf}")
            continue

        with lf.open("r", encoding="utf-8") as f:
            for line in f:
                p = line.strip().strip('"')
                if not p:
                    continue
                img = Path(p)
                lab = _img_to_label_path(img)
                if not lab.exists():
                    # 可能是背景图（无 label），允许
                    missing_label += 1
                    scanned += 1
                    if scanned >= max_images:
                        break
                    continue

                try:
                    txt = lab.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    scanned += 1
                    if scanned >= max_images:
                        break
                    continue

                if txt:
                    for row in txt.splitlines():
                        row = row.strip()
                        if not row:
                            continue
                        tok = row.split()
                        if not tok:
                            continue
                        try:
                            cid = int(float(tok[0]))
                            class_ids.add(cid)
                        except Exception:
                            pass

                scanned += 1
                if scanned >= max_images:
                    break

        if scanned >= max_images:
            break

    ok = (class_ids.issubset({0}) or len(class_ids) == 0)
    _log(f"[CHECK] scanned_images={scanned}, missing_label(background?)={missing_label}, class_ids={sorted(class_ids) if class_ids else []}")
    return ok, class_ids


def get_best_weights_path(run_name: str) -> Path:
    return RUNS_DIR / run_name / "weights" / "best.pt"


def do_train(stage_cfg: dict, data_yaml: Path, init_weights: Optional[Path]) -> Path:
    """
    用 Ultralytics Python API 训练（不需要命令行参数）
    """
    from ultralytics import YOLO  # type: ignore

    run_name = stage_cfg["name"]
    _log("\n" + "=" * 90)
    _log(f"[TRAIN] {run_name}")
    _log(f"  data   : {data_yaml}")
    _log(f"  model  : {MODEL_YAML}")
    _log(f"  init_w : {init_weights if init_weights else '(none)'}")
    _log(f"  imgsz  : {stage_cfg['imgsz']}  batch={stage_cfg['batch']}  epochs={stage_cfg['epochs']}  patience={stage_cfg['patience']}")
    _log("=" * 90)

    # 确保结构来自 YAML，然后再 load 权重（等价于你 CLI 里的 --model + --weights）
    model = YOLO(str(MODEL_YAML))
    if init_weights is not None and init_weights.exists():
        model.load(str(init_weights))

    model.train(
        data=str(data_yaml),
        imgsz=int(stage_cfg["imgsz"]),
        batch=int(stage_cfg["batch"]),
        epochs=int(stage_cfg["epochs"]),
        patience=int(stage_cfg["patience"]),
        device=DEVICE,
        workers=int(WORKERS),
        amp=bool(AMP),

        optimizer=str(stage_cfg["optimizer"]),
        lr0=float(stage_cfg["lr0"]),
        lrf=float(stage_cfg["lrf"]),
        cos_lr=bool(stage_cfg["cos_lr"]),
        weight_decay=float(stage_cfg["weight_decay"]),

        mosaic=float(stage_cfg["mosaic"]),
        close_mosaic=int(stage_cfg["close_mosaic"]),
        mixup=float(stage_cfg["mixup"]),
        copy_paste=float(stage_cfg["copy_paste"]),

        project=str(RUNS_DIR),
        name=str(run_name),
        exist_ok=False,

        seed=int(SEED),
        deterministic=True,
        verbose=True,
    )

    best = get_best_weights_path(run_name)
    if not best.exists():
        raise RuntimeError(f"best.pt not found after training: {best}")

    _log(f"[OK] best weights: {best}")
    return best


def do_val(weights: Path, data_yaml: Path, split: str = "val", imgsz: int = 960, name_suffix: str = "VAL") -> None:
    """
    用 Ultralytics Python API 评估
    """
    from ultralytics import YOLO  # type: ignore

    _log("\n" + "-" * 90)
    _log(f"[EVAL-{name_suffix}] weights={weights.name}  data={data_yaml.name}  split={split}  imgsz={imgsz}")
    _log("-" * 90)

    model = YOLO(str(weights))
    r = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(imgsz),
        device=DEVICE,
        workers=int(WORKERS),
        max_det=300,
        iou=0.7,   # NMS iou（与你日志一致）
        conf=None  # mAP 会扫阈值，不要固定 conf
    )

    # r 里有很多字段，不同版本不完全一致，这里只做稳健打印
    try:
        mp = getattr(r, "box").mp  # type: ignore
        mr = getattr(r, "box").mr  # type: ignore
        map50 = getattr(r, "box").map50  # type: ignore
        map5095 = getattr(r, "box").map  # type: ignore
        _log(f"[METRIC] P={mp:.4f}  R={mr:.4f}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
    except Exception:
        _log("[METRIC] (could not parse metrics object; check ultralytics version print above)")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    t0 = time.time()
    _log(f"[ROOT] {ROOT}")

    # 基础检查
    for p in [ROOT, MODEL_YAML, SF1_PY, SRC_YAML, INIT_WEIGHTS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    set_reproducible(SEED)
    ensure_allin_holdout_yaml()
    register_sf1_modules()

    # 单类检查（用 holdout lists 抽检，避免你再去找 tools\check_label_classes.py）
    train_list = HOLDOUT_DIR / "train_all.txt"
    val_list   = HOLDOUT_DIR / "val_holdout.txt"
    _log("\n[STEP] quick check label class ids ...")
    ok, cids = check_single_class_from_lists([train_list, val_list], max_images=8000)
    if not ok:
        _log(f"[FATAL] found non-zero class ids: {sorted(cids)}")
        _log("请先统一所有标签 class=0（单类），否则训练会异常。")
        sys.exit(2)

    # -------------------------
    # Stage A
    # -------------------------
    bestA = do_train(STAGE_A, HOLDOUT_YAML, INIT_WEIGHTS)
    do_val(bestA, HOLDOUT_YAML, split="val", imgsz=STAGE_A["imgsz"], name_suffix="HOLDOUT")

    # （可选）在原始 test split 上做一次“参考评估”
    if EVAL_ON_SRC_TEST:
        do_val(bestA, SRC_YAML, split="test", imgsz=832, name_suffix="SRC_TEST_REF")

    # -------------------------
    # Stage B
    # -------------------------
    bestB = do_train(STAGE_B, HOLDOUT_YAML, bestA)
    do_val(bestB, HOLDOUT_YAML, split="val", imgsz=STAGE_B["imgsz"], name_suffix="HOLDOUT")

    if EVAL_ON_SRC_TEST:
        do_val(bestB, SRC_YAML, split="test", imgsz=832, name_suffix="SRC_TEST_REF")

    # 总结
    _log("\n" + "=" * 90)
    _log("[DONE] FINAL TRAINING COMPLETE")
    _log(f"  StageA best: {bestA}")
    _log(f"  StageB best: {bestB}")
    _log(f"  Time used  : {(time.time() - t0) / 3600:.3f} h")
    _log("=" * 90)


if __name__ == "__main__":
    main()
