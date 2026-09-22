# 初学者导读

## 先掌握三个区别

- **逻辑块与物理块**：逻辑块是一个请求里的第几个块；物理块是池中的槽位。不同请求可以把逻辑块映射到同一个物理块。
- **引用归零与立即删除**：没有请求使用的完整块仍可保留在缓存中，只有容量不足时才淘汰。
- **哈希相等与内容相等**：哈希用于缩小查找范围，最终必须比较 token。否则一次碰撞就可能拿到另一个请求的 K/V。

## 建议的阅读顺序

1. 运行 `examples/basic.cpp`，手画两个请求的 block table，记录每个物理块的 refs。
2. 看公共头文件中的 `BlockHandle`、`BlockTable`、`Slot`。理解 `none` 是无效下标，不能用于访问数组。
3. 读构造函数与 `allocate/release`，先只考虑未发布块，追踪 free list 的头指针。
4. 读 `append_idle/unlink_idle`，分别画出空链表、单节点、删除头/尾/中间节点的指针变化。
5. 读 `publish/lookup`，理解哈希桶为什么保存多个槽位，以及完整前缀为何不可省略。
6. 读 `acquire`，把每次分配和引用增长与异常分支中的释放配对。
7. 读测试，特别是容量不足回滚、强制哈希碰撞和随机引用模型。

## 如何写有用的注释

注释应解释“为什么”，以及修改代码时必须保持的约定。例如 `++refs` 的注释不应只是“加一”，而应说明“另一个请求开始使用该块，所以淘汰逻辑必须看到新增引用”。

遇到链表修改，要解释节点当前属于哪条链表；遇到异常处理，要解释哪些状态已经改变、哪些引用需要回滚；遇到 public API，要说明返回的是借用句柄还是新增所有权。这些信息比逐行翻译 C++ 更有价值。

本项目的实现注释集中在这些位置：generation 防旧句柄、LRU 只容纳零引用缓存、publish 的异常顺序、lookup 的碰撞验证，以及 acquire 构造不完整时逐个回滚。

## 测试在验证什么

`tests/block_manager_test.cpp` 包含 5 组检查：分配与引用、前缀与位置查询、LRU 与碰撞、异常回滚与输入检查、固定种子的 3000 步随机引用模型。测试不用 `assert`，因此 Release 构建也会真正执行验证。

随机测试将所有存活请求表中的引用独立计数，然后和 manager 的 refs 对比；它能发现“某个异常路径漏释放了一次”的问题。强制哈希恒为 0 则让所有前缀落入同一桶，确保正确性不依赖哈希碰巧没有碰撞。

可选的内存与未定义行为检查（GCC/Clang）：

```bash
cmake -S . -B build-sanitize -DKVFLUX_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-sanitize -j 2
ctest --test-dir build-sanitize --output-on-failure
```

## 可以动手尝试的练习

把示例容量改为 3，观察两个不同尾块为什么导致容量不足；把前缀改成只有最后一个完整块相同，验证不能共享；先释放再重新 acquire 同一前缀，观察 generation 不变；制造淘汰后再分配，观察 generation 递增。

完成这些练习后，再尝试增加统计命中率或实现 RAII 表包装器。不要直接给共享尾块添加写接口：在引入 append 前，需要先设计 copy-on-write 和真实数据就绪状态。

## 继续学习 GPU 版本

先读 `kv_layout.h/.cpp` 的多维布局与容量公式，再读 `gpu_memory_pool.h/.cpp` 的一次分配、设备切换和数据就绪状态。运行 `kvflux_gpu_demo` 观察 block 17 的真实设备地址，最后阅读 `gpu_memory_pool_test.cpp` 的逐字节比较。完整命令见 [GPU 内存池说明](gpu_memory_pool.md)。
