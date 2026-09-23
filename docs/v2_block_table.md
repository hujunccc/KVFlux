# v2 Milestone 2：Block Table

Block table 回答一个问题：**序列中的第 i 个逻辑块，放在哪个物理页？** 逻辑块没有独立对象；vector 的下标 `i` 就是 logical block ID。物理页由 [Physical Block Pool](v2_physical_block_pool.md) 分配。

```text
Sequence A 的逻辑下标       0    1   2   3
对应的物理编号             27   91   3  48
BlockTable(A)             [27, 91, 3, 48]
```

`table[0]` 返回 `27`，`table[2]` 返回 `3`；`table.physical_id(i)` 是等价的具名写法。逻辑编号只表达**在本序列中的位置**；物理编号只表达**在池中的槽位**。因此物理编号不必连续，也不必等于逻辑编号。将来 paged attention 可以按逻辑下标查表，再到 GPU KV 存储中访问对应物理页。

## 接口与引用

v0 已有 `kvflux::BlockTable`，用于 token/prefix 教学接口。v2 使用 `kvflux::v2::BlockTable`，避免改变 v0/v1 行为。

表内部存放带 `generation` 的 `PhysicalBlockHandle`，对外查询返回 `PhysicalBlockID`。这样表看起来是 `[27, 91, 3, 48]`，同时能借助池校验旧句柄。每个表项持有一个引用：

- `append_new()`：向池申请一页，追加为下一逻辑块；返回物理编号。
- `append_existing(handle)`：已有持有者继续保留其引用，表为该页增加一个引用。
- `append_shared(source, i)`：把另一张表中逻辑块 `i` 的物理页共享到当前表尾；两张表必须使用同一个池。
- `table[i]` / `physical_id(i)`：读取逻辑块 `i` 对应的物理编号，不增加引用；越界会抛 `std::out_of_range`。
- `pop_back()` / `clear()` / 析构：移除映射并归还表持有的引用。

因此表可以移动，不能普通复制；普通复制若不增加引用，会让两张表在析构时重复归还同一份所有权。池必须比使用它的表活得更久。`append_existing` 的调用者仍负责归还自己的原始引用。共享不会复制 K/V，也不会再占一个物理页。GPU 仍在读取某页时，调度层必须等计算完成后才能释放最后一个引用。

最小用法如下；作用域结束时两张表会自动归还各自持有的引用：

```cpp
kvflux::PhysicalBlockPool pool(1024, 16);
kvflux::v2::BlockTable a(pool);
a.append_new();                 // logical block 0 → physical block 0
a.append_new();                 // logical block 1 → physical block 1
kvflux::v2::BlockTable b(pool);
b.append_shared(a, 0);         // b[0] 与 a[0] 指向同一物理页
auto physical = b[0];          // 0；读取编号不会增加引用
```

当前表只管理整块映射；它不保存 token 数、不负责尾块填充，也不处理请求调度。将来需要按 token 访问时，可用 `logical_block = token_index / block_size`，`offset = token_index % block_size`，再以 `physical_id(logical_block)` 定位物理页。实际 GPU 地址换算属于后续的 `PagedKVStorage`。

## 运行

```bash
cmake -S . -B build -DKVFLUX_BUILD_TESTS=ON
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/kvflux_block_table_demo
```

示例输出包含 `0 → 27`、`1 → 91`、`2 → 3`、`3 → 48`；表清空后，100 个物理页重新可用。测试还检查共享引用、容量不足、非法下标和移动后的所有权。
