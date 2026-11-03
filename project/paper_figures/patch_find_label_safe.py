from pathlib import Path
import shutil

script = Path(r"D:\YOLO-WTB\paper_figures\make_figure8_tpfpfn.py")

backup = script.with_suffix(".py.bak_before_find_label_fix")
shutil.copy2(script, backup)

text = script.read_text(encoding="utf-8")

start = text.find("def find_label(image_path):")
if start < 0:
    raise RuntimeError("没有找到 def find_label(image_path):")

end = text.find("\ndef load_gt_boxes", start)
if end < 0:
    end = text.find("\ndef ", start + 1)

if end < 0:
    raise RuntimeError("没有找到 find_label() 后面的下一个函数，无法安全替换。")

new_find_label = '''def find_label(image_path):
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

    # 4. 已确认存在的标签目录
    fixed_label_dirs = [
        r"D:\\YOLO-WTB\\test_data\\test\\labels",
        r"D:\\YOLO-WTB\\test_data2\\test\\labels",
        r"D:\\YOLO-WTB\\test_data_test_fix_v1\\test\\labels",
        r"D:\\YOLO-WTB\\test_data_train_fix_v1\\test\\labels",
        r"D:\\YOLO-WTB\\1\\test_pick\\wtmd\\labels",
        r"D:\\YOLO-WTB\\1\\test_pick\\yolov12n\\labels",
        r"D:\\YOLO-WTB\\scale_eval\\wtmd\\labels",
        r"D:\\YOLO-WTB\\scale_eval\\wtmd2\\labels",
        r"D:\\YOLO-WTB\\scale_eval_fixed\\wtmd\\labels",
        r"D:\\YOLO-WTB\\scale_eval_fixed\\wtmd2\\labels",
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
    search_root = Path(os.environ.get("WTB_LABEL_SEARCH_ROOT", r"D:\\YOLO-WTB"))

    if search_root.exists():
        matches = sorted(
            [p for p in search_root.rglob(stem + ".txt") if p.is_file()],
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
        f"未找到标签文件: {stem}.txt\\n"
        f"图像路径: {image_path}\\n"
        f"已搜索根目录: {search_root}"
    )
'''

text2 = text[:start] + new_find_label + "\n\n" + text[end+1:]

script.write_text(text2, encoding="utf-8")

print(f"[OK] find_label() 已安全替换")
print(f"[BACKUP] {backup}")
