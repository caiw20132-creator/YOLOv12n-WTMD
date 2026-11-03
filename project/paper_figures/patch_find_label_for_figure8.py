from pathlib import Path
import re

script = Path(r"D:\YOLO-WTB\paper_figures\make_figure8_tpfpfn.py")
text = script.read_text(encoding="utf-8")

new_find_label = r'''def find_label(image_path):
    import os
    from pathlib import Path

    image_path = Path(image_path)
    stem = image_path.stem
    parts = list(image_path.parts)

    label_candidates = []

    # 1. 图片同级目录下的 labels 文件夹
    label_candidates.append(image_path.parent / "labels" / (stem + ".txt"))

    # 2. 标准 YOLO 结构：.../images/... -> .../labels/...
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            new_parts = parts.copy()
            new_parts[idx] = "labels"
            label_candidates.append(Path(*new_parts).with_suffix(".txt"))

    # 3. 上一级目录下的 labels 文件夹
    label_candidates.append(image_path.parent.parent / "labels" / (stem + ".txt"))

    # 4. 明确加入已确认存在的测试集标签目录
    fixed_label_dirs = [
        r"D:\YOLO-WTB\test_data\test\labels",
        r"D:\YOLO-WTB\test_data2\test\labels",
        r"D:\YOLO-WTB\test_data_test_fix_v1\test\labels",
        r"D:\YOLO-WTB\1\test_pick\wtmd\labels",
        r"D:\YOLO-WTB\1\test_pick\yolov12n\labels",
        r"D:\YOLO-WTB\scale_eval\wtmd\labels",
        r"D:\YOLO-WTB\scale_eval_fixed\wtmd\labels",
    ]

    for d in fixed_label_dirs:
        label_candidates.append(Path(d) / (stem + ".txt"))

    # 5. 图片同目录同名 txt
    label_candidates.append(image_path.with_suffix(".txt"))

    for p in label_candidates:
        if p.exists():
            print(f"[LABEL] {stem} -> {p}")
            return p

    # 6. 全局兜底搜索
    search_root = Path(os.environ.get("WTB_LABEL_SEARCH_ROOT", r"D:\YOLO-WTB"))

    if search_root.exists():
        matches = []
        for p in search_root.rglob(stem + ".txt"):
            if p.is_file():
                matches.append(p)

        matches = sorted(
            matches,
            key=lambda x: (
                0 if "labels" in [part.lower() for part in x.parts] else 1,
                0 if "test_data" in str(x).lower() else 1,
                len(str(x))
            )
        )

        if matches:
            print(f"[LABEL] {stem} -> {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        f"未找到标签文件: {stem}.txt\n"
        f"图像路径: {image_path}\n"
        f"已搜索根目录: {search_root}"
    )
'''

text2 = re.sub(
    r"def find_label\(image_path\):.*?\n(?=def load_gt_boxes)",
    new_find_label + "\n\n",
    text,
    flags=re.S
)

if text2 == text:
    raise RuntimeError("find_label() 替换失败，请检查 make_figure8_tpfpfn.py 中函数名是否变化。")

script.write_text(text2, encoding="utf-8")
print("[OK] find_label() 已修复")
