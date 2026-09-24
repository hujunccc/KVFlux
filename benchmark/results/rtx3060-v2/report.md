# KVFlux V2：RTX 3060 Paged KV 性能验收

测量时间（UTC）：2026-09-24T17:18:45Z。GPU：NVIDIA GeForce RTX 3060 Laptop GPU，compute capability 8.6，CUDA runtime 12040，driver API 13020。

配置：单层 FP32 K/V，block size 16，2 KV heads、4 query heads、head size 32。每种配置预热 5 次，逐次测量 30 次。计时排除 fixture 创建、K/V 初始化和正确性回读；计入每次公开 API 的页表上传、临时分配、kernel 与同步。
`write_one_token` 反复覆盖已分配的最后一个 slot；`gather_full_kv` 读取预填充的完整 KV；两个 attention 项只计 attention API，均不包含 request 创建、append 或写入新 KV。

## Host 端完整调用延迟

单位为微秒；下表是 30 次样本的中位数。`fragmented` 表示请求的物理块编号全部为互不相邻的奇数页。

| 操作 | 长度 | Batch | 连续页 | 碎片页 | 碎片页 / 连续页 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `write_one_token` | 64 | 1 | 10.00 | 9.83 | 0.983× |
| `write_one_token` | 256 | 1 | 9.48 | 9.63 | 1.016× |
| `write_one_token` | 1024 | 1 | 9.60 | 9.36 | 0.975× |
| `gather_full_kv` | 64 | 1 | 9.94 | 9.75 | 0.981× |
| `gather_full_kv` | 256 | 1 | 9.78 | 9.92 | 1.015× |
| `gather_full_kv` | 1024 | 1 | 11.75 | 11.26 | 0.958× |
| `decode_attention` | 64 | 1 | 397.36 | 374.08 | 0.941× |
| `decode_attention` | 256 | 1 | 1465.14 | 1464.42 | 1.000× |
| `decode_attention` | 1024 | 1 | 5004.49 | 4995.03 | 0.998× |
| `batch_decode_attention` | 256 | 4 | 2475.07 | 2525.31 | 1.020× |
| `batch_decode_attention` | 256 | 16 | 2536.90 | 2538.28 | 1.001× |

## CUPTI GPU 活动记录

每项在 256-token 碎片页配置下采集 10 次；batch decode 使用 16 个请求。表中是每次调用的 GPU kernel 与元数据 H2D copy 时长中位数，属于 CUPTI Activity 时间戳，和上表未启用 profiler 的 host 样本分别采集；CUPTI 插桩会扰动时间，不能把两张表的时长直接相减。

| 操作 | Kernel 中位数 (µs) | 元数据拷贝中位数 (µs) | 每次元数据字节数 |
| --- | ---: | ---: | ---: |
| `write_one_token` | 1.82 | 0.38 | 8 |
| `gather_full_kv` | 2.22 | 0.38 | 128 |
| `decode_attention` | 1457.20 | 0.38 | 128 |
| `batch_decode_attention` | 2935.61 | 0.86 | 2176 |

## 观察与边界

- 单请求 decode attention 从 64 到 1024 token，host 中位延迟由 397.36 µs 增至 5004.49 µs（12.6×）。该简化 kernel 按 token 顺序扫描，并对每个输出维度重复 QK 点积；CUPTI 的 attention kernel 时间远大于页表 H2D 时间，主要瓶颈在计算 kernel。
- 256-token、batch 16 的一次 decode 调用为 2536.90 µs，折合 158.56 µs/请求。与 16 次单请求调用的中位数之和相比，估算吞吐提升 9.2×；这是同配置微基准推算，不是模型吞吐。
- 这些样本中，碎片页没有出现稳定的显著延迟惩罚；不应将此推广到其他长度、GPU 架构或模型。写入与 gather 的 host 延迟包含每次调用的分配、上传和同步，不能直接解释成纯 kernel 时间。
- CUPTI 记录了 kernel、memcpy 时间及页表字节数；未采集 SM occupancy、cache hit rate、DRAM 吞吐等硬件计数器。GPU 频率、温度和其他进程可能改变结果。

## 复现

```bash
cmake -S . -B build-v2-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_BENCHMARKS=ON \
  -DKVFLUX_BUILD_TESTS=ON -DKVFLUX_BUILD_EXAMPLES=OFF
cmake --build build-v2-release -j 4
ctest --test-dir build-v2-release --output-on-failure
./build-v2-release/benchmark/kvflux_v2_benchmark benchmark/results/v2-local
./build-v2-release/benchmark/kvflux_v2_cupti_profile benchmark/results/v2-local
python3 scripts/analyze_v2_benchmark.py benchmark/results/v2-local
```

CUPTI target 仅在 CMake 找到 `cupti.h` 和 `libcupti` 时创建。仓库中的 [samples.csv](samples.csv)、[summary.csv](summary.csv)、[cupti_activity.csv](cupti_activity.csv) 和 [environment.txt](environment.txt) 保留原始证据。
