#!/usr/bin/env python3
"""Summarize V2 host latency samples and CUPTI GPU activity into one report.

只依赖 Python 标准库；原始 CSV 保持不变，report.md 可反复生成。
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_v2_benchmark.py <result_directory>")
    directory = Path(sys.argv[1])
    environment = dict(
        line.split("=", 1)
        for line in (directory / "environment.txt").read_text(encoding="utf-8").splitlines()
    )
    samples = rows(directory / "samples.csv")
    summary = rows(directory / "summary.csv")
    activity = rows(directory / "cupti_activity.csv")

    # 防止旧 summary 与新的原始样本不一致；报告始终基于实际样本重新核算。
    groups = defaultdict(list)
    for row in samples:
        groups[(row["operation"], int(row["tokens"]), int(row["batch"]), row["layout"])].append(
            float(row["latency_us"])
        )
    for row in summary:
        key = (row["operation"], int(row["tokens"]), int(row["batch"]), row["layout"])
        measured = groups[key]
        if len(measured) != int(environment["measured_calls"]):
            raise ValueError(f"wrong sample count for {key}")
        if abs(statistics.median(measured) - float(row["median_us"])) > 0.02:
            raise ValueError(f"summary median does not match samples for {key}")

    def median(operation, tokens, batch, layout):
        return statistics.median(groups[(operation, tokens, batch, layout)])

    activity_groups = defaultdict(list)
    for row in activity:
        activity_groups[(row["operation"], int(row["sample"]))].append(row)
    profile = {}
    for operation in {key[0] for key in activity_groups}:
        captures = [records for (name, _), records in activity_groups.items() if name == operation]
        if len(captures) != 10:
            raise ValueError(f"wrong CUPTI capture count for {operation}")
        kernels = []
        copies = []
        bytes_uploaded = []
        for records in captures:
            kernel = [float(record["duration_us"]) for record in records if record["kind"] == "kernel"]
            memcpy = [record for record in records if record["kind"] == "memcpy"]
            if len(kernel) != 1:
                raise ValueError(f"expected one kernel for {operation}")
            kernels.append(kernel[0])
            copies.append(sum(float(record["duration_us"]) for record in memcpy))
            bytes_uploaded.append(sum(int(record["bytes"]) for record in memcpy))
        if len(set(bytes_uploaded)) != 1:
            raise ValueError(f"inconsistent metadata bytes for {operation}")
        profile[operation] = (
            statistics.median(kernels), statistics.median(copies), bytes_uploaded[0]
        )

    lines = [
        "# KVFlux V2：RTX 3060 Paged KV 性能验收",
        "",
        f"测量时间（UTC）：{environment['utc']}。GPU：{environment['gpu']}，"
        f"compute capability {environment['compute_capability']}，"
        f"CUDA runtime {environment['runtime']}，driver API {environment['driver']}。",
        "",
        f"配置：单层 FP32 K/V，block size {environment['block_size']}，"
        f"{environment['kv_heads']} KV heads、{environment['query_heads']} query heads、"
        f"head size {environment['head_size']}。每种配置预热 {environment['warmup_calls']} 次，"
        f"逐次测量 {environment['measured_calls']} 次。计时排除 fixture 创建、K/V 初始化和正确性回读；"
        "计入每次公开 API 的页表上传、临时分配、kernel 与同步。",
        "`write_one_token` 反复覆盖已分配的最后一个 slot；`gather_full_kv` 读取预填充的完整 KV；"
        "两个 attention 项只计 attention API，均不包含 request 创建、append 或写入新 KV。",
        "",
        "## Host 端完整调用延迟",
        "",
        "单位为微秒；下表是 30 次样本的中位数。`fragmented` 表示请求的物理块编号全部为互不相邻的奇数页。",
        "",
        "| 操作 | 长度 | Batch | 连续页 | 碎片页 | 碎片页 / 连续页 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for operation, token_counts, batches in (
        ("write_one_token", (64, 256, 1024), (1,)),
        ("gather_full_kv", (64, 256, 1024), (1,)),
        ("decode_attention", (64, 256, 1024), (1,)),
        ("batch_decode_attention", (256,), (4, 16)),
    ):
        for tokens in token_counts:
            for batch in batches:
                contiguous = median(operation, tokens, batch, "contiguous")
                fragmented = median(operation, tokens, batch, "fragmented")
                lines.append(
                    f"| `{operation}` | {tokens} | {batch} | {contiguous:.2f} | "
                    f"{fragmented:.2f} | {fragmented / contiguous:.3f}× |"
                )

    lines += [
        "",
        "## CUPTI GPU 活动记录",
        "",
        "每项在 256-token 碎片页配置下采集 10 次；batch decode 使用 16 个请求。"
        "表中是每次调用的 GPU kernel 与元数据 H2D copy 时长中位数，属于 CUPTI Activity 时间戳，"
        "和上表未启用 profiler 的 host 样本分别采集；CUPTI 插桩会扰动时间，"
        "不能把两张表的时长直接相减。",
        "",
        "| 操作 | Kernel 中位数 (µs) | 元数据拷贝中位数 (µs) | 每次元数据字节数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for operation in ("write_one_token", "gather_full_kv", "decode_attention", "batch_decode_attention"):
        kernel, copy, bytes_count = profile[operation]
        lines.append(f"| `{operation}` | {kernel:.2f} | {copy:.2f} | {bytes_count} |")

    single_64 = median("decode_attention", 64, 1, "contiguous")
    single_256 = median("decode_attention", 256, 1, "contiguous")
    single_1024 = median("decode_attention", 1024, 1, "contiguous")
    batch_16 = median("batch_decode_attention", 256, 16, "contiguous")
    throughput_gain = 16 * single_256 / batch_16
    lines += [
        "",
        "## 观察与边界",
        "",
        f"- 单请求 decode attention 从 64 到 1024 token，host 中位延迟由 "
        f"{single_64:.2f} µs 增至 {single_1024:.2f} µs（{single_1024 / single_64:.1f}×）。"
        "该简化 kernel 按 token 顺序扫描，并对每个输出维度重复 QK 点积；"
        "CUPTI 的 attention kernel 时间远大于页表 H2D 时间，主要瓶颈在计算 kernel。",
        f"- 256-token、batch 16 的一次 decode 调用为 {batch_16:.2f} µs，"
        f"折合 {batch_16 / 16:.2f} µs/请求。与 16 次单请求调用的中位数之和相比，"
        f"估算吞吐提升 {throughput_gain:.1f}×；这是同配置微基准推算，不是模型吞吐。",
        "- 这些样本中，碎片页没有出现稳定的显著延迟惩罚；不应将此推广到其他长度、"
        "GPU 架构或模型。写入与 gather 的 host 延迟包含每次调用的分配、上传和同步，"
        "不能直接解释成纯 kernel 时间。",
        "- CUPTI 记录了 kernel、memcpy 时间及页表字节数；未采集 SM occupancy、"
        "cache hit rate、DRAM 吞吐等硬件计数器。GPU 频率、温度和其他进程可能改变结果。",
        "",
        "## 复现",
        "",
        "```bash",
        "cmake -S . -B build-v2-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \\",
        "  -DCMAKE_CUDA_ARCHITECTURES=86 -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_BENCHMARKS=ON \\",
        "  -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_BUILD_EXAMPLES=OFF",
        "cmake --build build-v2-release -j 4",
        "ctest --test-dir build-v2-release --output-on-failure",
        "./build-v2-release/benchmark/kvflux_v2_benchmark benchmark/results/v2-local",
        "./build-v2-release/benchmark/kvflux_v2_cupti_profile benchmark/results/v2-local",
        "python3 scripts/analyze_v2_benchmark.py benchmark/results/v2-local",
        "```",
        "",
        "CUPTI target 仅在 CMake 找到 `cupti.h` 和 `libcupti` 时创建。"
        "仓库中的 [samples.csv](samples.csv)、[summary.csv](summary.csv)、"
        "[cupti_activity.csv](cupti_activity.csv) 和 [environment.txt](environment.txt) 保留原始证据。",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(directory / "report.md")


if __name__ == "__main__":
    main()
