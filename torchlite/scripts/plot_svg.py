#!/usr/bin/env python3
import csv
import math
import os
import sys


def read_history(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_acc": float(row["test_acc"]),
                "test_loss": float(row["test_loss"]),
            })
    return rows


def svg_line_plot(path, series, title, x_label, y_label, width=900, height=480):
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    x_vals = [x for x, _ in series[0][1]]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(min(y for _, y in s[1]) for s in series)
    y_max = max(max(y for _, y in s[1]) for s in series)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def x_to_px(x):
        return margin + (x - x_min) * plot_w / (x_max - x_min)

    def y_to_px(y):
        return margin + plot_h - (y - y_min) * plot_h / (y_max - y_min)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>',
        f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="12" font-family="sans-serif">{x_label}</text>',
        f'<text x="15" y="{height/2}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 15,{height/2})">{y_label}</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1"/>',
    ]

    for idx, (label, points) in enumerate(series):
        color = colors[idx % len(colors)]
        path_d = " ".join(
            [
                ("M" if i == 0 else "L") + f"{x_to_px(x):.2f},{y_to_px(y):.2f}"
                for i, (x, y) in enumerate(points)
            ]
        )
        parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(
            f'<text x="{margin + 10}" y="{margin + 18 + idx*18}" font-size="12" font-family="sans-serif" fill="{color}">{label}</text>'
        )

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def plot_grokking(csv_path, out_dir):
    rows = read_history(csv_path)
    if not rows:
        raise SystemExit("No rows in history CSV")
    epochs = [r["epoch"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    test_acc = [r["test_acc"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    test_loss = [r["test_loss"] for r in rows]

    svg_line_plot(
        os.path.join(out_dir, "grokking_accuracy.svg"),
        [("train_acc", list(zip(epochs, train_acc))), ("test_acc", list(zip(epochs, test_acc)))],
        "Grokking Accuracy",
        "epoch",
        "accuracy",
    )
    svg_line_plot(
        os.path.join(out_dir, "grokking_loss.svg"),
        [("train_loss", list(zip(epochs, train_loss))), ("test_loss", list(zip(epochs, test_loss)))],
        "Grokking Loss",
        "epoch",
        "loss",
    )


def plot_sin_cos(out_dir):
    points = [(x, math.sin(x), math.cos(x)) for x in [i * 0.1 for i in range(0, 629)]]
    xs = [p[0] for p in points]
    sin_vals = [p[1] for p in points]
    cos_vals = [p[2] for p in points]
    svg_line_plot(
        os.path.join(out_dir, "sin_cos.svg"),
        [("sin", list(zip(xs, sin_vals))), ("cos", list(zip(xs, cos_vals)))],
        "sin/cos",
        "x",
        "value",
    )


def main():
    if len(sys.argv) < 2:
        print("usage: plot_svg.py <grokking.csv> [out_dir]", file=sys.stderr)
        raise SystemExit(1)
    csv_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plot_grokking(csv_path, out_dir)
    plot_sin_cos(out_dir)
    print("wrote:")
    print(os.path.join(out_dir, "grokking_accuracy.svg"))
    print(os.path.join(out_dir, "grokking_loss.svg"))
    print(os.path.join(out_dir, "sin_cos.svg"))


if __name__ == "__main__":
    main()
