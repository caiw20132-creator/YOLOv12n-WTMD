from pathlib import Path
import sys, math, csv, argparse
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\YOLO-WTB")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve_path(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()

def load_val_images(data_yaml: Path):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    yaml_dir = data_yaml.parent
    base = yaml_dir
    if "path" in cfg and cfg["path"] not in [None, ""]:
        base = resolve_path(yaml_dir, str(cfg["path"]))
    val = cfg["val"]
    val_path = resolve_path(base, str(val))

    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        imgs = []
        for line in val_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                if not p.is_absolute():
                    p = (val_path.parent / p).resolve()
                imgs.append(p)
        return imgs

    if val_path.is_dir():
        imgs = [p for p in val_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        imgs.sort()
        return imgs

    raise FileNotFoundError(f"Cannot resolve val set from {val_path}")

def img_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        s = str(img_path)
        s = s.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
        return Path(s).with_suffix(".txt")

def yolo_xywhn_to_xyxy(xc, yc, w, h, iw, ih):
    x1 = (xc - w / 2.0) * iw
    y1 = (yc - h / 2.0) * ih
    x2 = (xc + w / 2.0) * iw
    y2 = (yc + h / 2.0) * ih
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def load_gt(img_path):
    with Image.open(img_path) as im:
        iw, ih = im.size
    lp = img_to_label_path(img_path)
    recs = []
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            ss = line.strip().split()
            if len(ss) < 5:
                continue
            cls_id = int(float(ss[0]))
            xc, yc, w, h = map(float, ss[1:5])
            xyxy = yolo_xywhn_to_xyxy(xc, yc, w, h, iw, ih)
            side = math.sqrt(max(w * 960, 0.0) * max(h * 960, 0.0))
            recs.append({"cls": cls_id, "box": xyxy, "side": side})
    return recs

def predict(model, img_path, imgsz=960, conf=0.001, iou=0.5, max_det=10000, device="0"):
    res = model.predict(
        source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        device=device, augment=False, verbose=False
    )[0]
    out = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, confs):
            out.append({"cls": int(c), "box": b.astype(np.float32), "conf": float(s)})
    return out

def match_stats(preds, gts, iou_thr=0.5):
    used = [False] * len(gts)
    tp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if used[j] or p["cls"] != g["cls"]:
                continue
            v = box_iou_xyxy(p["box"], g["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp += 1
    fp = max(len(preds) - tp, 0)
    fn = max(len(gts) - tp, 0)
    tiny_gt = sum(1 for g in gts if g["side"] < 32)
    return tp, fp, fn, tiny_gt

def draw_boxes(im, boxes, color, label_prefix=""):
    dr = ImageDraw.Draw(im)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(float, b["box"])
        dr.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = f"{label_prefix}{b.get('conf', ''):.2f}" if "conf" in b else label_prefix
        if txt.strip():
            dr.text((x1, max(0, y1 - 16)), txt, fill=color)
    return im

def crop_zoom(im, gts):
    if not gts:
        return im.crop((0, 0, min(256, im.width), min(256, im.height))).resize((256,256))
    g = sorted(gts, key=lambda x: x["side"])[0]
    x1, y1, x2, y2 = g["box"]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = max((x2 - x1) * 6, 120)
    h = max((y2 - y1) * 6, 120)
    l = max(int(cx - w / 2), 0)
    t = max(int(cy - h / 2), 0)
    r = min(int(cx + w / 2), im.width)
    b = min(int(cy + h / 2), im.height)
    return im.crop((l, t, r, b)).resize((256, 256))

def make_panel(img_path, base_model, ours_model, out_dir):
    raw = Image.open(img_path).convert("RGB")
    gts = load_gt(img_path)
    pb = predict(base_model, img_path)
    po = predict(ours_model, img_path)

    im0 = raw.copy()
    im1 = draw_boxes(raw.copy(), gts, (0,255,0), "")
    im2 = draw_boxes(raw.copy(), pb, (255,0,0), "")
    im3 = draw_boxes(raw.copy(), po, (0,0,255), "")

    zoom0 = crop_zoom(im0, gts)
    zoom1 = crop_zoom(im1, gts)
    zoom2 = crop_zoom(im2, gts)
    zoom3 = crop_zoom(im3, gts)

    W, H = 360, 360
    ims = [x.resize((W,H)) for x in [im0, im1, im2, im3]]
    labels = ["Original", "GT", "YOLOv12n", "YOLO12n-WTMD"]

    canvas = Image.new("RGB", (W*4, H+280), "white")
    dr = ImageDraw.Draw(canvas)

    for i, (im, lab) in enumerate(zip(ims, labels)):
        x = i * W
        canvas.paste(im, (x, 0))
        dr.text((x+10, H+8), lab, fill="black")
    for i, zm in enumerate([zoom0, zoom1, zoom2, zoom3]):
        x = i * W + 50
        canvas.paste(zm, (x, H+40))
        dr.rectangle([x, H+40, x+256, H+296], outline="black", width=2)

    stem = Path(img_path).stem
    out_path = out_dir / f"{stem}_4col_zoom.png"
    canvas.save(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-weights", required=True)
    ap.add_argument("--ours-weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--picked-txt", default="")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    val_images = load_val_images(Path(args.data))
    base_model = YOLO(args.base_weights)
    ours_model = YOLO(args.ours_weights)

    rows = []
    for idx, img_path in enumerate(val_images, 1):
        gts = load_gt(img_path)
        pb = predict(base_model, img_path)
        po = predict(ours_model, img_path)
        tb, fpb, fnb, tiny_gt = match_stats(pb, gts)
        to, fpo, fno, _ = match_stats(po, gts)
        rows.append({
            "image": str(img_path),
            "stem": img_path.stem,
            "gt": len(gts),
            "tiny_gt": tiny_gt,
            "base_tp": tb, "base_fp": fpb, "base_fn": fnb,
            "ours_tp": to, "ours_fp": fpo, "ours_fn": fno,
            "delta_tp": to - tb,
            "delta_fn": fnb - fno,
            "delta_fp": fpb - fpo
        })
        if idx % 100 == 0 or idx == len(val_images):
            print(f"[PROGRESS] {idx}/{len(val_images)}")

    cand_csv = outdir / "qual_candidates.csv"
    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["delta_tp"], x["delta_fn"], x["tiny_gt"]), reverse=True))
    print("[DONE]", cand_csv)

    if args.picked_txt:
        picked = []
        for line in Path(args.picked_txt).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                picked.append(s)
        fig_dir = outdir / "picked_panels"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for stem in picked:
            hits = [r for r in rows if r["stem"] == stem]
            if not hits:
                continue
            make_panel(Path(hits[0]["image"]), base_model, ours_model, fig_dir)

if __name__ == "__main__":
    main()
