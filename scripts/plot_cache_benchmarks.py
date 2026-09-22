#!/usr/bin/env python3
"""验证 v1 A/B/C 原始样本，输出中位数 CSV、图表和实验报告。"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ("workload", "trace", "compute_iterations", "lookahead")
COLORS = ["#5a6679", "#087e8b", "#c08038"]


def load(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def aggregate(directory):
    groups = defaultdict(list)
    for raw in load(directory / "cache_samples.csv"):
        row = {k: v if k in ("workload", "trace") else float(v) for k, v in raw.items()}
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise ValueError("Non-finite metric")
        assert row["gpu_blocks_total"] == row["gpu_blocks_used"] + row["gpu_blocks_free"]
        assert row["requests"] == row["gpu_demand_hits"] + row["prefetch_misses"]
        assert row["prefetch_count"] == sum(row[k] for k in ("prefetch_hits", "prefetch_late", "prefetch_unused", "prefetch_outstanding"))
        assert row["gpu_to_cpu_bytes"] == row["offload_count"] * row["block_bytes"]
        assert row["cpu_to_gpu_bytes"] == row["load_count"] * row["block_bytes"]
        assert row["transferring_blocks"] == 0
        row["traffic_mib"] = (row["gpu_to_cpu_bytes"] + row["cpu_to_gpu_bytes"]) / 2**20
        row["migrations_per_request"] = (row["offload_count"] + row["load_count"]) / row["requests"]
        row["demand_miss_rate"] = row["prefetch_misses"] / row["requests"]
        row["transfer_device_ms"] = row["gpu_to_cpu_ms"] + row["cpu_to_gpu_ms"]
        row["transfer_time_fraction"] = row["transfer_device_ms"] / row["total_ms"]
        groups[tuple(row[k] for k in KEYS)].append(row)
    expected = {(w, trace, iterations, window) for w in "ABC" for trace in ("locality", "scan")
                for iterations in (0, 500000) for window in (0, 1, 4)}
    if groups.keys() != expected:
        raise ValueError("Incomplete workload matrix")
    summaries = []
    for key, group in sorted(groups.items()):
        if len(group) != 5 or {r["sample"] for r in group} != set(range(5)):
            raise ValueError(f"Expected five unique samples for {key}")
        summary = dict(zip(KEYS, key))
        summary.update({k: median(r[k] for r in group) for k in group[0] if k not in KEYS + ("sample",)})
        summary["samples"] = len(group)
        summary["total_min_ms"] = min(r["total_ms"] for r in group)
        summary["total_max_ms"] = max(r["total_ms"] for r in group)
        summaries.append(summary)
    with (directory / "cache_summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def save(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def plots(directory, find):
    for trace in ("locality", "scan"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for axes_row, compute in zip(axes, (0, 500000)):
            for ax, metric, label in zip(axes_row, ("request_stall_ms", "total_ms"), ("Request stall", "Total runtime")):
                for offset, window, color in zip((-0.25, 0, 0.25), (0, 1, 4), COLORS):
                    values = [find(w, trace, compute, window)[metric] for w in "ABC"]
                    ax.bar([i + offset for i in range(3)], values, 0.24, color=color,
                           label="No prefetch" if not window else f"Next {window}")
                ax.set_xticks(range(3), ("A: 100/100", "B: 80/100", "C: 30/100"))
                ax.set_ylabel("ms / trace")
                ax.set_title(f"{label} | compute={compute}")
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(f"KVFlux v1 | {trace} | 1 MiB/block | median of 5 samples")
        save(fig, directory, trace)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, trace in zip(axes, ("locality", "scan")):
        for offset, window, color in zip((-0.25, 0, 0.25), (0, 1, 4), COLORS):
            values = [find(w, trace, 0, window)["migrations_per_request"] for w in "ABC"]
            ax.bar([i + offset for i in range(3)], values, 0.24, color=color,
                   label="No prefetch" if not window else f"Next {window}")
        ax.set_xticks(range(3), ("A: 100/100", "B: 80/100", "C: 30/100"))
        ax.set_title(f"Migration amplification | {trace}")
        ax.set_ylabel("(H2D + D2H blocks) / demand")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, directory, "thrashing")


def report(directory, summaries, find):
    lines = ["# KVFlux v1 最终实验：Metrics 与 A/B/C Benchmark Suite", "",
             "[环境](environment.txt) · [原始 180 个样本](cache_samples.csv) · [汇总 CSV](cache_summary.csv)", "",
             "固定 100 个逻辑块、每块 1 MiB，A/B/C 的 GPU 容量分别为 100/80/30 块。CPU 为全部 100 块预留 pinned backing。",
             "每个配置预热一次，再轮换容量与预取模式顺序采样 5 次。表中每个指标单独取中位数，原始数据保留波动。",
             "初始化及最终逐字节校验不计入时间/计数增量；每次样本均检查数据、容量和 metrics 守恒。", "",
             "## 主实验：局部性与容量", "",
             "`([0..59] × 3 + [60..99]) × 3`，共 660 次访问；60 块热集合在 B 中放得下，在 C 中放不下。",
             "A/B/C 使用完全相同的访问序列、数据和计算量，区别只有 GPU 容量。", "",
             "![局部性实验](locality.png)", "",
             "| Workload | Compute | Prefetch | Total ms | Stall ms | Offloads | Reloads | Traffic MiB | Hit / Miss / Late |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in summaries:
        if r["trace"] != "locality":
            continue
        lines.append(f"| {r['workload']} | {r['compute_iterations']:.0f} | {r['lookahead']:.0f} | {r['total_ms']:.2f} | "
                     f"{r['request_stall_ms']:.2f} | {r['offload_count']:.0f} | {r['load_count']:.0f} | {r['traffic_mib']:.0f} | "
                     f"{r['prefetch_hits']:.0f} / {r['prefetch_misses']:.0f} / {r['prefetch_late']:.0f} |")
    lines.extend(["", "## 迁移开销与压力", "",
                  "以下使用无计算、无预取的相同局部性轨迹，减去 A 的总耗时作为额外开销。它包含迁移、元数据和同步，不能解释为纯 PCIe 时间。", "",
                  "| Workload | Total ms | 相对 A 额外 ms | 迁移/访问 | D2H µs/block | H2D µs/block | D2H GB/s | H2D GB/s | Event time / total |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    a = find("A", "locality", 0, 0)
    for w in "ABC":
        r = find(w, "locality", 0, 0)
        lines.append(f"| {w} | {r['total_ms']:.2f} | {r['total_ms'] - a['total_ms']:.2f} | {r['migrations_per_request']:.3f} | "
                     f"{r['gpu_to_cpu_latency_us']:.2f} | {r['cpu_to_gpu_latency_us']:.2f} | {r['gpu_to_cpu_bandwidth_gbps']:.2f} | "
                     f"{r['cpu_to_gpu_bandwidth_gbps']:.2f} | {r['transfer_time_fraction']:.1%} |")
    lines.extend(["", "传输 latency 是 CUDA event 区间平均值，带宽为完成字节数 / 累计 event 时间，GB/s 使用十进制。",
                  "无迁移时输出 0，表示没有传输样本。Event 区间可能包含主机提交间隙；其占比不是 PCIe 利用率，也不能证明物理链路已饱和。", "",
                  "## Thrashing 对照：循环扫描", "",
                  "`[0..99] × 3`，共 300 次访问。循环工作集比 GPU 容量大时，LRU 在 80/100 下也可能每次 miss。",
                  "此对照防止把容量比例直接当作命中率；压力表现还依赖复用距离。", "",
                  "![扫描实验](scan.png)", "", "![迁移放大](thrashing.png)", "",
                  "| Workload | Prefetch | Offloads | Reloads | MiB | 迁移/访问 | 无计算总耗时 ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for w in "ABC":
        for window in (0, 1, 4):
            r = find(w, "scan", 0, window)
            lines.append(f"| {w} | {window} | {r['offload_count']:.0f} | {r['load_count']:.0f} | {r['traffic_mib']:.0f} | "
                         f"{r['migrations_per_request']:.3f} | {r['total_ms']:.2f} |")
    lines.extend(["", "## 预取是否有意义", ""])
    for w in "BC":
        base = find(w, "locality", 500000, 0)
        pf = find(w, "locality", 500000, 1)
        reduction = 1 - pf["request_stall_ms"] / base["request_stall_ms"]
        lines.append(f"- {w}、有计算窗口、next-1：stall {base['request_stall_ms']:.2f} → {pf['request_stall_ms']:.2f} ms（降低 {reduction:.1%}），"
                     f"total {base['total_ms']:.2f} → {pf['total_ms']:.2f} ms；流量 {base['traffic_mib']:.0f} → {pf['traffic_mib']:.0f} MiB。")
    lines.extend(["", "预取可以把部分传输等待移进计算窗口，但不能自动消除容量不足造成的来回迁移。无计算窗口的结果用于观察调度成本和批量等待。",
                  "计算为独立 scratch 上的合成 kernel，不是 attention；表中的 request 是一次 block demand，不是完整 LLM 请求。",
                  "Prefetch hit 只计一次及时使用；miss 是 demand 到来时未就绪，late 是 miss 的子集；普通 GPU 命中不算 prefetch hit。",
                  "P95、scheduler CPU time、unused、各方向字节数和最终容量均保存在 CSV。", "",
                  "## 传输与重叠基线", "",
                  "同次版本的 pageable/pinned/staging、sync/async overlap、batch 实验见 [传输报告](transfer/report.md)。",
                  "[v1 范围与六项验收](../../../docs/v1_completion.md) · [Metrics 定义与复现](../../../docs/metrics_benchmarks.md)", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    summaries = aggregate(directory)
    indexed = {tuple(r[k] for k in KEYS): r for r in summaries}
    def find(w, trace, compute, window):
        return indexed[w, trace, compute, window]
    plots(directory, find)
    report(directory, summaries, find)
    print(f"Validated 180 samples; wrote summary, plots and report to {directory}")


if __name__ == "__main__":
    main()
