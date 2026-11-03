from pathlib import Path
import csv

ROOT = Path(
    r"D:\YOLO-WTB\runs_controlled_hierarchy_minimal\verification"
)

overall_csv = (
    ROOT /
    "fixed_test_overall_baseline_vs_h11.csv"
)

scale_csv = (
    ROOT /
    "scale_conditioned_baseline_vs_h11.csv"
)

out_tex = (
    ROOT /
    "Table8_controlled_hierarchy.tex"
)


# ============================================================
# Read overall results
# ============================================================

with overall_csv.open(
    "r",
    encoding="utf-8-sig"
) as f:

    overall = list(
        csv.DictReader(f)
    )


# ============================================================
# Read scale results
# ============================================================

with scale_csv.open(
    "r",
    encoding="utf-8-sig"
) as f:

    scale = list(
        csv.DictReader(f)
    )


baseline = overall[0]
h11 = overall[1]


def get_scale(
    model,
    group
):
    for row in scale:

        if (
            row["Model"] == model
            and
            row["Scale_Group"] == group
        ):
            return row

    raise RuntimeError(
        f"Missing {model}, {group}"
    )


groups = [
    "tiny",
    "small",
    "medium-to-large"
]


# ============================================================
# Build LaTeX
# ============================================================

lines = []

lines.append(r"\begin{table}[H]")
lines.append(r"\centering")
lines.append(
    r"\caption{Controlled fixed-test evaluation of "
    r"prediction-hierarchy reallocation under a fixed "
    r"three-output budget. Panel A reports overall detection "
    r"performance, while Panel B reports scale-conditioned AP "
    r"under the primary long-side grouping.}"
)
lines.append(
    r"\label{8}"
)
lines.append(
    r"\renewcommand{\arraystretch}{1.15}"
)
lines.append(
    r"\setlength{\tabcolsep}{5pt}"
)
lines.append("")

# ============================================================
# Panel A
# ============================================================

lines.append(
    r"\textbf{Panel A: Overall fixed-test comparison}"
)
lines.append("")

lines.append(
    r"\begin{tabularx}{\textwidth}{"
    r">{\hsize=1.25\hsize\raggedright\arraybackslash}X "
    r">{\hsize=0.75\hsize\centering\arraybackslash}X "
    r"c c c c c}"
)

lines.append(r"\toprule")

lines.append(
    r"\textbf{Configuration} & "
    r"\textbf{Hierarchy} & "
    r"\textbf{$K$} & "
    r"\textbf{Precision} & "
    r"\textbf{Recall} & "
    r"\textbf{AP$_{50}$} & "
    r"\textbf{AP$_{50:95}$} \\"
)

lines.append(r"\midrule")

lines.append(
    "YOLOv12n baseline"
    " & P3/P4/P5"
    " & 3"
    f" & {float(baseline['Precision']):.4f}"
    f" & {float(baseline['Recall']):.4f}"
    f" & {float(baseline['AP50']):.4f}"
    f" & {float(baseline['AP50_95']):.4f}"
    r" \\"
)

lines.append(
    "Hierarchy-only reallocation"
    " & P2/P3/P4"
    " & 3"
    f" & {float(h11['Precision']):.4f}"
    f" & {float(h11['Recall']):.4f}"
    f" & {float(h11['AP50']):.4f}"
    f" & {float(h11['AP50_95']):.4f}"
    r" \\"
)

lines.append(r"\midrule")

lines.append(
    "Difference"
    " & --"
    " & --"
    f" & {float(h11['Precision']) - float(baseline['Precision']):+.4f}"
    f" & {float(h11['Recall']) - float(baseline['Recall']):+.4f}"
    f" & {float(h11['AP50']) - float(baseline['AP50']):+.4f}"
    f" & {float(h11['AP50_95']) - float(baseline['AP50_95']):+.4f}"
    r" \\"
)

lines.append(r"\bottomrule")
lines.append(r"\end{tabularx}")
lines.append("")
lines.append(r"\vspace{0.7em}")

# ============================================================
# Panel B
# ============================================================

lines.append(
    r"\textbf{Panel B: Scale-conditioned fixed-test comparison}"
)
lines.append("")

lines.append(
    r"\begin{tabularx}{\textwidth}{"
    r">{\hsize=0.90\hsize\centering\arraybackslash}X "
    r">{\hsize=1.25\hsize\centering\arraybackslash}X "
    r"c c c}"
)

lines.append(r"\toprule")

lines.append(
    r"\textbf{Hierarchy} & "
    r"\textbf{Scale group} & "
    r"\textbf{GT count} & "
    r"\textbf{AP$_{50}$} & "
    r"\textbf{AP$_{50:95}$} \\"
)

lines.append(r"\midrule")


for hierarchy, model in [
    ("P3/P4/P5", "YOLOv12n_Baseline"),
    ("P2/P3/P4", "H11_Reallocation"),
]:

    rows = [
        get_scale(
            model,
            group
        )
        for group in groups
    ]

    for i, row in enumerate(rows):

        hierarchy_text = (
            hierarchy
            if i == 0
            else ""
        )

        lines.append(
            f"{hierarchy_text}"
            f" & {row['Scale_Group']}"
            f" & {int(row['GT_Count'])}"
            f" & {float(row['AP50']):.4f}"
            f" & {float(row['AP50_95']):.4f}"
            r" \\"
        )

    if hierarchy == "P3/P4/P5":
        lines.append(r"\midrule")


lines.append(r"\bottomrule")
lines.append(r"\end{tabularx}")
lines.append("")
lines.append(r"\end{table}")


out_tex.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("=" * 80)
print("LATEX TABLE CREATED")
print("=" * 80)

print(out_tex)

print("\n" + "\n".join(lines))
