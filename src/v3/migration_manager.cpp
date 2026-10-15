#include "kvflux/v3/migration_manager.h"

#include "../cuda_utils.h"

#include <algorithm>
#include <stdexcept>

namespace kvflux::v3 {

MigrationManager::MigrationManager(CacheDirectory& directory, GpuChunkStore& gpu_store,
                                   HostChunkBuffers& cpu_buffers, ChunkFileStore& store,
                                   std::size_t io_threads)
    : directory_(directory), gpu_store_(gpu_store), cpu_buffers_(cpu_buffers),
      store_(store), transfer_stream_(gpu_store.device()) {
    if (gpu_store_.chunk_bytes() != cpu_buffers_.chunk_bytes()) {
        throw std::invalid_argument("GPU and CPU chunk sizes must match");
    }
    if (io_threads == 0) throw std::invalid_argument("io_threads must be positive");
    workers_.reserve(io_threads);
    for (std::size_t i = 0; i < io_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

MigrationManager::~MigrationManager() noexcept {
    // 收敛所有未完成任务：取消并等待，保证缓冲区晚于底层操作回收。
    for (auto& entry : tasks_) {
        auto& task = *entry.second;
        const auto state = task.state.load();
        if (state == MigrationState::QUEUED || state == MigrationState::RUNNING) {
            cancel(task.id);
        }
    }
    for (auto& entry : tasks_) {
        try { wait(entry.first); } catch (...) {}
    }
    {
        std::lock_guard<std::mutex> lock(io_mu_);
        io_stop_ = true;
    }
    io_work_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    for (auto& entry : tasks_) {
        if (entry.second->done) cudaEventDestroy(entry.second->done);
    }
}

TaskID MigrationManager::submit(const CacheKey& key, StorageTier destination,
                                std::uint64_t valid_tokens, int priority) {
    const auto& record = directory_.at(key);
    const auto& replica_of = [&](StorageTier t) -> const std::optional<Replica>& {
        switch (t) {
            case StorageTier::GPU: return record.gpu;
            case StorageTier::CPU: return record.cpu;
            case StorageTier::NVME: return record.nvme;
        }
        throw std::invalid_argument("unknown storage tier");
    };
    StorageTier source = StorageTier::GPU;
    bool found = false;
    for (const auto t : {StorageTier::GPU, StorageTier::CPU, StorageTier::NVME}) {
        if (t == destination) continue;
        // NVMe<->GPU 无直达路径，这样的副本不参与源推断。
        if ((t == StorageTier::GPU && destination == StorageTier::NVME) ||
            (t == StorageTier::NVME && destination == StorageTier::GPU)) {
            continue;
        }
        const auto& replica = replica_of(t);
        if (replica.has_value() && replica->state == ChunkState::READY) {
            if (found) throw std::logic_error("ambiguous source: multiple READY replicas");
            source = t;
            found = true;
        }
    }
    if (!found) throw std::logic_error("no READY source replica for migration");
    const auto& destination_replica = replica_of(destination);
    if (destination_replica.has_value()) {
        throw std::logic_error("destination tier already holds a replica");
    }
    if ((source == StorageTier::GPU && destination == StorageTier::NVME) ||
        (source == StorageTier::NVME && destination == StorageTier::GPU)) {
        throw std::invalid_argument("NVMe<->GPU migration must go through CPU");
    }

    auto task = std::make_shared<Task>();
    task->id = next_id_++;
    task->key = key;
    task->source = source;
    task->destination = destination;
    task->priority = priority;
    task->source_loc = replica_of(source)->location;
    task->meta = make_file_meta(key, valid_tokens, cpu_buffers_.chunk_bytes());

    const auto destination_state =
        destination == StorageTier::GPU ? ChunkState::LOADING : ChunkState::COPYING;
    directory_.set_state(key, source, ChunkState::COPYING);
    try {
        task->destination_loc = directory_.place(key, destination, destination_state);
        directory_.pin(key);
        if (task->uses_cuda()) {
            void* const dst = destination == StorageTier::GPU
                                  ? gpu_store_.device_ptr(task->destination_loc)
                                  : cpu_buffers_.data(task->destination_loc.slot);
            const void* const src = source == StorageTier::GPU
                                        ? gpu_store_.device_ptr(task->source_loc)
                                        : cpu_buffers_.data(task->source_loc.slot);
            detail::check(cudaEventCreateWithFlags(&task->done, cudaEventDisableTiming),
                          "cudaEventCreateWithFlags");
            detail::check(cudaMemcpyAsync(dst, src, gpu_store_.chunk_bytes(),
                                          cudaMemcpyDefault,
                                          transfer_stream_.native_handle()),
                          "cudaMemcpyAsync");
            detail::check(cudaEventRecord(task->done, transfer_stream_.native_handle()),
                          "cudaEventRecord");
            task->state = MigrationState::RUNNING;
        } else {
            {
                std::lock_guard<std::mutex> lock(io_mu_);
                io_queue_.push_back(task);
            }
            io_work_cv_.notify_one();
        }
    } catch (...) {
        rollback_submit(*task);
        throw;
    }
    const auto id = task->id;
    tasks_.emplace(id, std::move(task));
    return id;
}

void MigrationManager::rollback_submit(Task& task) {
    if (task.done) {
        cudaEventDestroy(task.done);
        task.done = nullptr;
    }
    // 若 memcpy 已提交但后续步骤失败，等 DMA 完成再回收槽位。
    transfer_stream_.synchronize();
    const auto& record = directory_.at(task.key);
    const bool placed =
        (task.destination == StorageTier::GPU && record.gpu.has_value()) ||
        (task.destination == StorageTier::CPU && record.cpu.has_value()) ||
        (task.destination == StorageTier::NVME && record.nvme.has_value());
    if (placed) {
        directory_.set_state(task.key, task.destination, ChunkState::ERROR);
        directory_.remove(task.key, task.destination);
        directory_.unpin(task.key);
    }
    directory_.set_state(task.key, task.source, ChunkState::READY);
}

void MigrationManager::finalize(Task& task) {
    if (task.cancel_requested.load()) {
        finalize_cancelled(task);
        return;
    }
    directory_.set_state(task.key, task.destination, ChunkState::READY);
    directory_.set_state(task.key, task.source, ChunkState::READY);
    directory_.unpin(task.key);
    task.state = MigrationState::COMPLETED;
}

void MigrationManager::finalize_cancelled(Task& task) {
    // 底层操作已确认完成（event 已同步或 I/O 线程已返回），缓冲区回收安全。
    if (task.destination == StorageTier::NVME &&
        store_.exists(task.destination_loc.slot)) {
        // 取消发生在写盘完成之后：已发布的文件一并删除，配额归还。
        store_.erase(task.destination_loc.slot);
    }
    directory_.set_state(task.key, task.destination, ChunkState::ERROR);
    directory_.remove(task.key, task.destination);
    directory_.set_state(task.key, task.source, ChunkState::READY);
    directory_.unpin(task.key);
    task.state = MigrationState::CANCELLED;
}

MigrationState MigrationManager::poll(TaskID id) {
    auto& task = find(id);
    const auto state = task.state.load();
    if (state != MigrationState::RUNNING) return state;

    bool ok = false;
    if (task.uses_cuda()) {
        const auto result = cudaEventQuery(task.done);
        if (result == cudaErrorNotReady) return MigrationState::RUNNING;
        if (result != cudaSuccess) {
            task.error = std::string("cudaEventQuery: ") + cudaGetErrorString(result);
        } else {
            ok = true;
        }
    } else {
        const int io = task.io_result.load();
        if (io == 0) return MigrationState::RUNNING;
        ok = io > 0;
    }
    if (!ok) {
        task.state = MigrationState::FAILED;
        directory_.set_state(task.key, task.destination, ChunkState::ERROR);
        directory_.remove(task.key, task.destination);
        directory_.set_state(task.key, task.source, ChunkState::READY);
        directory_.unpin(task.key);
        return task.state;
    }
    finalize(task);
    return task.state;
}

MigrationState MigrationManager::wait(TaskID id) {
    auto& task = find(id);
    for (;;) {
        const auto state = task.state.load();
        if (state == MigrationState::COMPLETED || state == MigrationState::FAILED ||
            state == MigrationState::CANCELLED) {
            return state;
        }
        if (state == MigrationState::QUEUED) {
            // I/O 线程还没接手；等它完成（或被 cancel 收走）。
            std::unique_lock<std::mutex> lock(io_mu_);
            io_done_cv_.wait(lock, [&] {
                return task.io_result.load() != 0 ||
                       task.state.load() != MigrationState::QUEUED;
            });
            continue;
        }
        if (task.uses_cuda()) {
            detail::check(cudaEventSynchronize(task.done), "cudaEventSynchronize");
        } else {
            std::unique_lock<std::mutex> lock(io_mu_);
            io_done_cv_.wait(lock, [&] { return task.io_result.load() != 0; });
        }
        return poll(id); // 完成路径统一走 poll 的 finalize。
    }
}

bool MigrationManager::cancel(TaskID id) {
    auto& task = find(id);
    for (;;) {
        const auto state = task.state.load();
        if (state == MigrationState::COMPLETED || state == MigrationState::FAILED ||
            state == MigrationState::CANCELLED) {
            return false;
        }
        task.cancel_requested = true;
        if (state == MigrationState::RUNNING) return true; // finalize 时丢弃目标副本
        // QUEUED：若 I/O 线程尚未取走，直接摘除并收尾；已取走则由完成路径处理。
        bool dequeued = false;
        {
            std::lock_guard<std::mutex> lock(io_mu_);
            const auto it = std::find(io_queue_.begin(), io_queue_.end(),
                                      tasks_.at(id));
            if (it != io_queue_.end()) {
                io_queue_.erase(it);
                dequeued = true;
            }
        }
        if (dequeued) {
            finalize_cancelled(task);
            io_done_cv_.notify_all();
            return true;
        }
        // 与 worker 取任务的竞态：重新观察状态再决定。
    }
}

MigrationTaskInfo MigrationManager::info(TaskID id) const {
    const auto it = tasks_.find(id);
    if (it == tasks_.end()) throw std::out_of_range("unknown migration task");
    const auto& task = *it->second;
    return MigrationTaskInfo{task.id, task.key, task.source, task.destination,
                             task.state.load(), task.priority, task.error};
}

std::size_t MigrationManager::pending_count() const {
    std::size_t n = 0;
    for (const auto& entry : tasks_) {
        const auto state = entry.second->state.load();
        if (state == MigrationState::QUEUED || state == MigrationState::RUNNING) ++n;
    }
    return n;
}

MigrationManager::Task& MigrationManager::find(TaskID id) const {
    const auto it = tasks_.find(id);
    if (it == tasks_.end()) throw std::out_of_range("unknown migration task");
    return *it->second;
}

void MigrationManager::worker_loop() {
    for (;;) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(io_mu_);
            io_work_cv_.wait(lock, [&] { return io_stop_ || !io_queue_.empty(); });
            if (io_stop_ && io_queue_.empty()) return;
            // 优先级高的先执行，平级按提交顺序。
            auto best = io_queue_.begin();
            for (auto it = io_queue_.begin(); it != io_queue_.end(); ++it) {
                if ((*it)->priority > (*best)->priority ||
                    ((*it)->priority == (*best)->priority && (*it)->id < (*best)->id)) {
                    best = it;
                }
            }
            task = std::move(*best);
            io_queue_.erase(best);
            task->state = MigrationState::RUNNING;
        }
        if (task->cancel_requested.load()) {
            task->io_result = 1; // I/O 未发生；owner 的 finalize 按取消收尾。
        } else {
            try {
                do_io(*task);
                task->io_result = 1;
            } catch (const std::exception& e) {
                task->error = e.what();
                task->io_result = -1;
            } catch (...) {
                task->error = "unknown io error";
                task->io_result = -1;
            }
        }
        io_done_cv_.notify_all();
    }
}

void MigrationManager::do_io(Task& task) {
    if (task.destination == StorageTier::NVME) {
        store_.write(task.destination_loc.slot, task.meta,
                     cpu_buffers_.data(task.source_loc.slot));
    } else {
        const bool found = store_.read(task.source_loc.slot, task.meta,
                                       cpu_buffers_.data(task.destination_loc.slot),
                                       cpu_buffers_.chunk_bytes());
        if (!found) throw ChunkMissError("nvme chunk file missing for registered replica");
    }
}

} // namespace kvflux::v3
