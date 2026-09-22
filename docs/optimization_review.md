# 当前实现的优化检查

这份检查覆盖 v0 管理器和 v1.0–v1.2 GPU 池。优先保证真实数据生命周期，再优化长序列开销。

## 本轮已经处理

| 问题 | 改动与效果 |
| --- | --- |
| 每次请求申请显存会引入分配开销 | 构造一次连续池，按 id 定位，请求只修改元数据 |
| total_blocks 间接调用 stats 会扫描池 | 增加 `BlockManager::capacity()`，总块数和地址查询保持 O(1) |
| 新块可能暴露上次使用者的数据 | 每次 allocate 重置初始化状态，完整 write 成功后才能 read/publish |
| 活跃共享块被覆盖会污染其他请求 | write 拒绝 refs>1 或已发布块 |
| 模型维度乘积溢出可能分配过小内存 | 布局和 pool 大小计算显式检查溢出 |
| 数据尚未就绪就进入前缀索引 | GPU pool 强制 write 完成之后 publish，不暴露模拟 acquire |

## 建议的后续顺序

### 1. 引用所有权改成 RAII

位置：`include/kvflux/block_manager.h` 的 `BlockTable`、`BlockHandle`，以及所有 release 调用。

现在普通复制不会 retain，忘记 release 会占住容量；错误重复 release 还可能减掉其他使用方的引用。generation 只能防槽位复用后的过期句柄，不能解决每份引用的所有权。

建议先引入不可复制、可移动的引用包装器，析构自动归还；显式 clone/share 新增引用，且用 manager 身份拒绝跨池句柄。需要同时明确 manager 必须比所有引用活得更久。验收应覆盖异常退出、移动赋值、重复释放与跨 manager 使用。

### 2. 去掉完整前缀的重复存储和计算

位置：`Slot::prefix`、`BlockManager::acquire/publish/lookup`。

长度为 T、块大小为 B 的序列逐块保留完整前缀，累计元数据及哈希工作约为 O(T²/B)。长上下文时 CPU 和 host 内存会先成为瓶颈。

建议采用共享父前缀节点及块级 token，递增计算前缀哈希。哈希碰撞仍必须验证完整上下文身份；不能为了提速取消碰撞检查。先用多个长度（如 128/512/2048/8192 tokens）、热命中/冷分配和强制碰撞场景建立基线，再比较优化后结果。

### 3. 优化表检查和统计

位置：`BlockManager::validate` 的双重循环、`stats()` 的全池扫描。

目前整表 share/release 的重复句柄检查是 O(N²)，remaining_blocks 也是 O(capacity)。建议用明确的表所有权减少对可信内部表的重复验证；对外部输入保留检查。也可维护状态计数器实现 O(1) 统计，但每个状态迁移都必须同步计数，并用随机模型验证。

### 4. 增加传输吞吐能力

位置：`GpuMemoryPool::write_block/read_block`。

v1.3–v1.5 已实现 pinned host buffer、异步 batch、compute/transfer streams，以及在途引用保护。实测说明 pinned staging 的 CPU 拷贝可能抵消收益，batch 也不保证线性提速。下一步优先减少 staging 拷贝、合并相邻块、使用双缓冲，再做细粒度完成事件回收。

### 5. 扩大工程验证与观测

v1.9–v1.10 已补齐容量/迁移/预取 metrics，以及 100/80/30 槽位的局部性和循环扫描实验，见 [最终报告](../benchmark/results/rtx3060-v1/report.md)。实验显示重度压力可达到每次 demand 两次迁移；预取降低 stall 但不减少流量。后续可补充跨长度基准、分配失败计数、CPU CI 和 GPU CI；合成基准不足以证明真实推理收益。后续接入模型后应验证 K/V 数值与推理结果一致，再评估 layout 是否需要按 layer 连续、对齐或向量化。

## 仍然明确保留的限制

单线程、手动引用管理、没有 append/copy-on-write、没有模型推理、尚未接入真实推理调度器。原始 device pointer 是底层逃生接口，外部直接写可以绕过发布保护；它应只交给了解引用和同步约定的后端代码。
