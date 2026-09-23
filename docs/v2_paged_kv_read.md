# v2 Milestone 8：Paged KV Read

写入后的 K/V 不按请求连续存放。控制面从 `SequenceState` 生成 `ReadBlockTable`；第 `i` 项是逻辑块 `i` 对应的物理块编号。`read_paged_kv` 将它上传到 GPU，kernel 对每个逻辑 token 查表，读取 K/V，并按请求 token 顺序输出连续缓冲，供后续 attention 计算使用。

```text
logical_token → logical_block = token / block_size → block_table[logical_block]
             → physical_block → K/V[physical_block][head][token % block_size][dim]
```

例如请求有 40 个 token，`block_size=16`，block table 为 `[27, 3, 91]`：token `0–15` 从 P27 读，`16–31` 从 P3 读，`32–39` 从 P91 读。尾块只读 8 个有效 token；物理页无需连续。输出 K/V 形状均为 `[40][num_kv_heads][head_size]`，数据按 dtype 原始位模式搬运，不做数值转换。

接口在 [paged_kv_read.h](../include/kvflux/v2/paged_kv_read.h)。空请求不访问输出；非空请求要求 K/V 输出均为存储所在 GPU 上足够大的独立设备缓冲。拒绝错误页池、空指针、错误设备指针及与缓存或彼此重叠的输出。调用方在读取完成前必须保持请求和物理页存活，且需保证页内容已经写入。当前接口每次调用上传 block table、使用默认 stream 并同步返回。

```bash
cmake -S . -B build-cuda -DKVFLUX_ENABLE_CUDA=ON -DKVFLUX_BUILD_TESTS=ON
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/kvflux_paged_kv_read_demo
```

CPU 测试验证 `[27, 3, 91]` 的逻辑映射；GPU 测试用 Milestone 7 写入 40 个 token 后，验证 FP16、BF16、FP32 的完整 K/V 读回、跨页边界、尾块和输出边界。这里提供 attention 可消费的连续读出路径；真正的 attention kernel、批量调度与读时免拷贝访问留待后续阶段。
