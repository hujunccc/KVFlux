#!/usr/bin/env python3
"""从基准 CSV 生成可导出的图和汇总；不替换或平滑原始测量值。"""
import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kvflux-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    transfer = rows(directory / "transfers.csv")
    overlap = rows(directory / "overlap.csv")
    batch = rows(directory / "batch.csv")
    sizes = sorted({int(row["size_bytes"]) for row in transfer})
    size_labels = ["4 KiB", "64 KiB", "1 MiB", "16 MiB", "256 MiB"]
    if sizes != [4096, 65536, 1048576, 16777216, 268435456]:
        raise ValueError("unexpected transfer size set")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"pageable": "#5a6679", "pinned_direct": "#087e8b", "pinned_staged": "#df8132"}
    labels = {"pageable": "Pageable", "pinned_direct": "Pinned direct", "pinned_staged": "Pinned + CPU staging"}
    for metric, name, ylabel in [("bandwidth_gbps", "bandwidth", "Bandwidth (GB/s)"),
                                 ("latency_us", "latency", "Latency (microseconds)")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, direction in zip(axes, ["H2D", "D2H"]):
            for mode in colors:
                selected = sorted((r for r in transfer if r["direction"] == direction and r["mode"] == mode),
                                  key=lambda r: int(r["size_bytes"]))
                ax.plot(sizes, [float(r[metric]) for r in selected], "o-", label=labels[mode], color=colors[mode])
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, size_labels)
            if name == "latency":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.set_title(direction)
            ax.set_xlabel("Transfer size")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=9)
        fig.suptitle("KVFlux | Median of 9 samples; allocation excluded")
        save(fig, directory, name)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    names = [r["mode"] for r in overlap]
    for ax, field, title in zip(axes, ["total_ms", "effective_bandwidth_gbps", "overlap_ratio"],
                               ["Total runtime (ms)", "Effective bandwidth (GB/s)", "Shorter interval overlapped (%)"]):
        values = [float(r[field]) * (100 if field == "overlap_ratio" else 1) for r in overlap]
        bars = ax.bar(names, values, color=["#5a6679", "#087e8b"])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_ylim(0, max(max(values) * 1.22, 1))
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("16 MiB H2D + independent fake compute")
    save(fig, directory, "overlap")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for mode, label, color in [("sync_per_block", "Sync per block", "#5a6679"),
                                ("async_batch", "Async batch", "#087e8b")]:
        selected = [r for r in batch if r["mode"] == mode]
        ax.plot([int(r["block_count"]) for r in selected], [float(r["bandwidth_gbps"]) for r in selected],
                "o-", label=label, color=color)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Blocks per batch (1 MiB/block)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("H2D | Queue N copies, synchronize once")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, directory, "batch")

    lines = ["# KVFlux 实测基准", "", "环境见 [environment.txt](environment.txt)。所有数值为 9 个样本的中位数；分配和预热不计时。", "",
             "![传输带宽](bandwidth.png)", "", "![传输延迟](latency.png)", "",
             "| Size | Direction | Mode | Latency (µs) | Bandwidth (GB/s) |",
             "| --- | --- | --- | ---: | ---: |"]
    for r in transfer:
        label = size_labels[sizes.index(int(r["size_bytes"]))]
        lines.append(f"| {label} | {r['direction']} | {r['mode']} | {float(r['latency_us']):.2f} | {float(r['bandwidth_gbps']):.3f} |")
    lines.extend(["", "![同步与异步](overlap.png)", "",
                  "| Mode | Total (ms) | Transfer (ms) | Compute (ms) | Effective GB/s | Overlap |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in overlap:
        lines.append(f"| {r['mode']} | {float(r['total_ms']):.3f} | {float(r['transfer_ms']):.3f} | {float(r['compute_ms']):.3f} | {float(r['effective_bandwidth_gbps']):.3f} | {float(r['overlap_ratio']):.1%} |")
    lines.extend(["", "Overlap 是两个 CUDA event 区间交集 / 较短区间长度，不是 GPU 利用率；effective bandwidth 使用包含计算的总 wall time。", "",
                  "![Batch 带宽](batch.png)", "",
                  "| Mode | Blocks | Total (µs) | Bandwidth (GB/s) |", "| --- | ---: | ---: | ---: |"])
    for r in batch:
        lines.append(f"| {r['mode']} | {r['block_count']} | {float(r['total_us']):.2f} | {float(r['bandwidth_gbps']):.3f} |")
    lines.extend(["", "CSV：[传输](transfers.csv)、[重叠](overlap.csv)、[批量](batch.csv)。同目录包含 SVG 矢量图，便于导出。",
                  "", "这些是本机微基准，不能外推成模型吞吐收益；host 内存/PCIe/时钟和工作负载会影响结果。", ""])
    (directory / "report.md").write_text("\n".join(lines))
    print(f"Wrote plots and report to {directory}")


if __name__ == "__main__":
    main()
