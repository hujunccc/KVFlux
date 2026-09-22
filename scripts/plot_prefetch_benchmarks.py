#!/usr/bin/env python3
"""将真实预取 CSV 汇总为报告和 PNG/SVG，不平滑或替换测量数据。"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def save(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    results = rows(directory / "prefetch.csv")
    pressure = rows(directory / "pressure.csv")
    workloads = sorted({int(row["compute_iterations"]) for row in results})
    fig, axes = plt.subplots(len(workloads), 2, figsize=(10, 4 * len(workloads)),
                             squeeze=False, layout="constrained")
    for axes_row, iterations in zip(axes, workloads):
        selected = [r for r in results if int(r["compute_iterations"]) == iterations]
        labels = ["No prefetch" if r["lookahead"] == "0" else f"Next {r['lookahead']}" for r in selected]
        for ax, field, title in zip(axes_row, ("request_stall_ms", "total_ms"),
                                    ("Request stall", "Total runtime")):
            values = [float(r[field]) for r in selected]
            bars = ax.bar(labels, values, color=["#5a6679", "#087e8b", "#399a71", "#c08038"])
            ax.bar_label(bars, fmt="%.2f", padding=3)
            ax.set_ylim(0, max(max(values) * 1.2, 1))
            ax.set_ylabel("ms / 36 accesses")
            ax.set_title(f"{title} | compute iterations={iterations}")
            ax.grid(axis="y", alpha=0.2)
    save(fig, directory, "prefetch")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    counts = [int(r["created_blocks"]) for r in pressure]
    for field, label in (("gpu_resident", "GPU resident"), ("cpu_resident", "CPU resident")):
        ax.plot(counts, [int(r[field]) for r in pressure], "o-", label=label)
    ax.axhline(8, linestyle="--", color="#888888", label="GPU capacity = 8")
    ax.set(xlabel="Logical blocks created", ylabel="Blocks", xticks=counts, ylim=(0, 9))
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, directory, "pressure")

    lines = ["# v1.6–v1.8：GPU/CPU 分层和预取实测", "",
             "环境见 [environment.txt](environment.txt)。每块 1 MiB，8 个 GPU 槽位、12 个逻辑块，顺序访问 3 遍。",
             "每个模式预热一次，轮换顺序测量 7 个样本；下表为各指标的样本中位数。", "",
             "![压力轨迹](pressure.png)", "",
             "创建 12 块后 GPU resident=8、CPU resident=4，LRU 搬出 4 块；迁移后逐字节校验通过。", "",
             "![等待时间与总耗时](prefetch.png)", "",
             "| Compute iterations | Lookahead | Stall (ms) | 相对无预取降低 | Total (ms) | P95 stall (µs) | Scheduler CPU (ms) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        baseline = next(x for x in results if x["compute_iterations"] == r["compute_iterations"] and x["lookahead"] == "0")
        reduction = 1 - float(r["request_stall_ms"]) / float(baseline["request_stall_ms"])
        lines.append(f"| {r['compute_iterations']} | {r['lookahead']} | {float(r['request_stall_ms']):.3f} | {reduction:.1%} | "
                     f"{float(r['total_ms']):.3f} | {float(r['p95_stall_us']):.2f} | {float(r['scheduler_cpu_ms']):.3f} |")
    lines.extend(["", "Stall 为 acquire_gpu 的 host wall time 之和，包含按需搬出/装入和接口开销；负的降低百分比表示变慢。",
                  "Total 包含预取提交、计算、轮询和尾部迁移，不能仅凭 stall 减少断言端到端加速。",
                  "Scheduler CPU 是 prefetch/poll 调用耗时之和，不包含全部 query 循环 CPU 开销。", "",
                  "计算使用独立 scratch 上的 fake compute，不执行 attention。compute=0 用于观察没有计算重叠窗口时的开销。",
                  "这是本机合成访问轨迹的实测；7 次样本不代表跨设备、跨负载的收益保证。", "",
                  "原始数据：[汇总](prefetch.csv)、[逐次样本及迁移计数](prefetch_samples.csv)、[压力轨迹](pressure.csv)。",
                  "同目录提供 SVG 图。复现命令和接口约定见 [分层缓存说明](../../../docs/tiered_cache.md)。", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote plots and report to {directory}")


if __name__ == "__main__":
    main()
