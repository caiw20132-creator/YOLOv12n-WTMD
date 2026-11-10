"""Validate packaged records without training, inference, or loading weights."""
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "metadata/file_sha256.json"
METRICS = ("precision", "recall", "ap50", "ap50_95")


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def verify():
    inventory = read_json(MANIFEST)
    paths = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    require(paths == set(inventory) | {MANIFEST}, "file inventory")
    for name, expected in inventory.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "SHA-256: " + name)

    checkpoints = read_json("metadata/checkpoints.json")
    present = {c["path"]: c["sha256"] for c in checkpoints if c["included"]}
    require(len(checkpoints) == 9 and len(present) == 4, "checkpoint count")
    require(set(present) == {n for n in paths if n.endswith(".pt")}, "weight payload inventory")
    require(all(inventory[path] == sha for path, sha in present.items()), "checkpoint SHA-256")

    members = rows("data/wtmd_image_label_manifest.csv")
    require(Counter(r["split"] for r in members) == {"train":4021,"val":1151,"test":572}, "split sizes")
    require(len({r["image_sha256"] for r in members}) == 5744, "current split image hashes")
    expected_counts = {"train": (17043,210), "val": (4791,50), "test": (2301,34)}
    for split, expected in expected_counts.items():
        sub = [r for r in members if r["split"] == split]
        require((sum(int(r["annotations"]) for r in sub), sum(r["label_state"] == "empty_background" for r in sub)) == expected,
                "annotations and backgrounds: " + split)

    index = read_json("data/reported_results/table_index.json")
    main = [t for t in index.values() if "final_pdf_table" in t]
    require(len(index) == 24 and len(main) == 14 and {t["final_pdf_table"] for t in main} == set(range(1,14)), "table coverage")
    for t in index.values():
        count = t.get("data_rows", t.get("rows_including_headers"))
        require(len(rows(t["file"])) == count, "table row count: " + t["file"])
    for item in read_json("metadata/experiment_index.json"):
        require(all(path in paths for path in item["files"]), "experiment index: " + item["analysis"])

    primary = {r["model"]: tuple(r[k] for k in METRICS) for r in rows(index["native_primary"]["file"])}
    console = (ROOT / "evidence/native_primary_console.txt").read_text(encoding="utf-8-sig")
    result = re.findall(r"(?m)^\[RESULT\] (YOLOv12n(?:-WTMD)?) \| P=(\S+) \| R=(\S+) \| AP50=(\S+) \| AP50:95=(\S+)", console)
    require(len(result) == 2 and "Test images: 572" in console and "Test instances counted from labels: 2301" in console, "native console result")
    for model, *values in result:
        require(primary[model] == tuple(f"{Decimal(v):.4f}" for v in values), "native result: " + model)

    external = rows("audit/external_model_assignment.csv")
    require(Counter(r["evidence_status"] for r in external) == {"model_assignment_conflict":4,"manuscript_summary_only":2}, "external evidence")
    for r in external:
        if r["evidence_status"] == "model_assignment_conflict":
            other = next(x for x in external if x["images"] == r["images"] and x["model"] != r["model"])
            require(all(Decimal(r["archive_"+k]) == Decimal(other["paper_"+k]) for k in METRICS), "archived model assignment")
    active_external = rows(index["external_native_901"]["file"])
    require(len(active_external) == 2 and all(r["images"] == "901" for r in active_external), "active external scope")

    nms = rows(index["s1_interventions_all_thresholds"]["file"])
    require(len({(r["removal"],r["level"],r["matching_iou"]) for r in nms}) == 60, "intervention record keys")
    for r in nms:
        require(int(r["reentered"]) == sum(int(r[k]) for k in ("unmatched","replacement","new_match")), "NMS re-entry balance")
        require(int(r["gained_gt"]) - int(r["lost_gt"]) == int(r["net_match"]), "NMS GT balance")
    print(json.dumps({"passed": True, "sha256_checked_files": len(inventory), "registered_weights_present": len(present),
                      "current_manifest_images": len(members), "main_pdf_tables": 13, "result_csv_files": len(index),
                      "nms_intervention_rows": len(nms)}, indent=2))


if __name__ == "__main__":
    verify()
