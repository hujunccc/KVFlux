# v3 Milestone 0：Tier Abstraction + Cache Directory

V3.0 是 V3 的元数据基础：**不进行真实数据传输**，只回答两个问题——
某份 KV 存在哪里，以及它是否能被安全使用。

## 数据单元：KVChunk

跨层存储和迁移的基本单位是 KVChunk，第一版 **1 chunk = 1 逻辑 KV block**。
每个 chunk 的身份明确记录覆盖的层范围 `[first_layer, last_layer)`，
单层版本取 `[l, l+1)`；不会出现 GPU 侧只复制一个 layer、而 CPU/NVMe 侧
将其解释为完整模型 KV 的情况。以后可扩展为一个 chunk 合并多个物理块，
减少小 I/O 开销。

## CacheKey：逻辑身份 ≠ 物理位置

`CacheKey` 是 chunk 的逻辑身份，与任何物理位置无关：

```text
KV_A：GPU Block 17 → CPU Buffer 8 → NVMe File Location 24
      CacheKey 始终不变，GPU BlockID 回收重用也不会影响它
```

除 token 前缀外，key 还包含 `model_id`、`layout_version`、`dtype` 和层范围，
避免不同配置的缓存被错误复用。`prefix_hash` 复用 v0 的 FNV-1a；prefix 原文
保留用于精确比较，哈希碰撞不会误命中。用 `make_cache_key` 由前缀构造，
保证哈希与原文一致。

## Tier 抽象

```cpp
enum class StorageTier { GPU, CPU, NVME };
```

三层都实现 `Tier` 接口：`allocate / release / capacity / available /
allocated_chunks / used_bytes / available_bytes`。每个槽位固定容纳一个
chunk；位置 `TierLocation{slot, generation}` 复用 v0 的 generation 校验，
旧位置在槽位重用后被拒绝。

| Tier | 实现 | 说明 |
| --- | --- | --- |
| GPU | 复用 V2 `PhysicalBlockPool` | slot 即 PhysicalBlockID |
| CPU | `SlotTier`（定长槽位池） | V1 pinned 组件依赖 CUDA；V3.0 不传数据，先用元数据占位 |
| NVMe | `SlotTier` | slot 是文件块编号；真实文件分配属 V3.2 |

## CacheDirectory

目录登记每份 KV 的：CacheKey、三级位置、**各层副本独立状态**、version、
ref count、pin count、last access time。

副本槽位只通过目录向对应 Tier 申请（`place`）和归还（`remove`），因此
不会产生重复或悬空引用：同层第二份副本抛 `logic_error`；有副本、引用或
钉住时 `destroy` 被拒绝。V3.1 起 `remove` 只接受 READY 或 ERROR 的副本，
传输进行中（COPYING/LOADING/EVICTING）的副本不能被移除，防止槽位在
复制完成前被回收。`stats(tier)` 同时给出层自身的容量/字节计数和
目录登记的副本数，两者一致。

chunk 身份与各层副本状态分开管理，同一 chunk 可以同时：

```text
GPU: READY    CPU: COPYING    NVMe: LOADING
```

`ChunkState` 取值 `READY / COPYING / LOADING / EVICTING / ERROR`，
`set_state` 只改指定层的副本，不影响其他层。纯位置迁移不改 `version`；
KV 内容变化时用 `bump_version` 递增。

## 验收与测试

测试模拟 5.6 流程：Chunk A 依次 GPU Block 17 → CPU Buffer 8 →
NVMe File Location 24，每步验证目录查询；另覆盖 BlockID 重用不影响
CacheKey、同层重复副本/悬空引用拒绝、三层副本独立状态、每层容量与
副本数统计、强制哈希碰撞不合并身份。全部可在无 GPU 的机器上运行。

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure -R 'kvflux_v3_(tier|cache_directory)_tests'
```

接口见 [tier.h](../include/kvflux/v3/tier.h) 与
[cache_directory.h](../include/kvflux/v3/cache_directory.h)。
