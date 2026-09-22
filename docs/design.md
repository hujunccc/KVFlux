# v0 设计与接口约定

## 数据结构与状态

`Slot` 是固定物理槽位的元数据。`id` 是槽位下标；`generation` 每次重新分配递增；`refs` 是使用方持有的引用数；`prefix` 和 `hash` 表示缓存身份。句柄 `{id,generation}` 能拒绝槽位复用之前的旧引用。

| 状态 | free | refs | cached | 所属链表 |
| --- | --- | --- | --- | --- |
| 未分配 | true | 0 | false | free list |
| 私有活跃 | false | 大于 0 | false | 无 |
| 已发布活跃 | false | 大于 0 | true | 无 |
| 缓存空闲 | false | 0 | true | LRU |

始终满足 `free + active + cached_idle == capacity`。active 统计物理块数，不是总引用数。

free list 是 `next_free` 串联的单向链表；头插归还、头取分配。LRU 是 `prev/next` 串联的双向链表；头部最旧，尾部最新。两种链表都自己维护，不依赖 `std::list`。

LRU 的“最近使用”定义为最近一次最后引用释放的时间顺序。命中空闲缓存时将其从 LRU 摘下，最后一个使用方释放时再放回尾部。活跃块不参与淘汰。缓存本身不持有引用，`refs==0` 不意味着内容已经消失。

## 两种 lookup

**前缀 lookup**：`lookup(prefix, out)` 通过 FNV-1a 哈希找到候选桶，再比较完整 token 前缀。命中时增加引用；调用方要负责释放。未命中返回 false，不修改 out。空前缀和非完整块前缀都不缓存。

**位置 lookup**：`block_at(table, token_index)` 计算 `token_index / block_size` 得到逻辑块下标，从表中取物理句柄；块内偏移是 `token_index % block_size`。返回的句柄是借用，不增加引用。v0 不提供实际字节地址，因为尚未定义层数、head 数、dtype 和 K/V 布局。

例如两个前缀 `[1,2,3,4]` 与 `[9,2,3,4]` 的最后两个 token 相同，但历史上下文不同，不能共享第二块。哈希包含完整前缀，而不是仅包含当前块。

## 生命周期 API

| 接口 | 成功后的所有权变化 |
| --- | --- |
| `allocate()` | 得到一个 refs=1 的未发布块；没有可回收槽位则抛 `CapacityError` |
| `retain(h)` | 增加一个引用；允许重新持有尚未淘汰的缓存块 |
| `release(h)` | 减少一个引用；归零后进入 free list 或 LRU |
| `publish(h, prefix)` | 私有块转换为可查找的完整缓存块，引用数不变 |
| `lookup(prefix, out)` | 命中增加一个引用；失败无变化 |
| `acquire(tokens)` | 返回持有每个逻辑块一个引用的表 |
| `share(table)` | 为整个表各增加一个引用，返回新的描述表 |
| `release(table)` | 释放表的引用并清空表；空表重复释放合法 |
| `block_at(table, index)` | 返回借用句柄，不增加引用 |

`publish` 要求 `refs==1`、尚未发布、前缀非空且长度能被 block size 整除。重复发布相同前缀会被拒绝，应先 lookup。缓存发布后在逻辑上不可修改；v0 没有数据写入 API，也没有 append 或 copy-on-write。

所有权采用手动约定：表的复制和句柄的复制不会调用 retain；也没有析构自动释放。不能把同一个所有权副本释放两次。generation 只防止槽位复用后的旧句柄，不是每次借用的唯一凭据，因此无法识别所有重复释放。例如仍有其他请求持有引用时，误释放一个已经释放过的裸句柄可能损坏对方计数。使用者必须成对管理 acquire/share/lookup/retain 与 release。

句柄只在所属 manager 内有效，不能跨 manager 传递。manager 不可复制或移动。单线程或调用者提供外部互斥；不做内部锁和异步事件管理。

## 错误处理与回滚

非法配置、过期句柄、重复发布、非法表使用 `std::invalid_argument`；token 下标越界使用 `std::out_of_range`；计数或 generation 耗尽使用 `std::overflow_error`。容量不足使用 `CapacityError`。内存分配失败等异常继续向上传播。

`acquire` 失败时释放本次已经取得的全部引用，包括从既有缓存命中的引用。它不恢复整个缓存到原始样子：已经淘汰的旧缓存不会恢复，已经发布的新缓存可以作为零引用缓存保留。保证的是无引用泄漏、既有活跃请求不被淘汰。

整表 release 先检查表长度、句柄、非零引用和重复句柄，再执行释放，避免输入损坏时发生部分释放。但无法检测手动复制造成的所有权错误。

## 复杂度与明确的取舍

设容量为 C，序列 token 数为 T，块大小为 B，块数为 N=ceil(T/B)，候选哈希桶大小为 K。

| 操作 | v0 复杂度 |
| --- | --- |
| free list 分配/归还、retain、LRU 插入/删除 | O(1) |
| 淘汰并移除前缀索引 | 平均 O(K)，桶内 vector 删除需要移动元素 |
| hash 长度 L 的前缀 | O(L) |
| lookup 长度 L 的前缀 | 平均 O(L)，恶意碰撞时 O(K×L) |
| acquire 全部前缀 | 通常 O(T + T²/B)，因逐块重新哈希/比较完整前缀 |
| share / release(table) | O(N²)，包含朴素的重复句柄检查 |
| block_at | O(1) |
| stats | O(C)，直接扫描统计 |

为方便初学者检查碰撞正确性，每个缓存块保存完整前缀。对于单条长序列，前缀元数据总量可达 O(T²/B)。固定的是槽位数，不是元数据字节数；本版不是严格固定内存或生产性能方案。

后续可使用共享前缀节点和增量哈希降低重复工作，同时必须保留碰撞验证机制。自定义 Hasher 应是确定性的纯函数，不能重入 manager；默认 FNV-1a 是非密码学哈希，不用于安全认证。
