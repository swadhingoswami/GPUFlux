#include "cuda_backend.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

inline double elapsed_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

inline ExecutionResult result_ok(double ms, uint64_t bytes) {
  ExecutionResult r = {ms, bytes, 1, 0};
  return r;
}
inline ExecutionResult result_fail() {
  ExecutionResult r = {0.0, 0, 0, 0};
  return r;
}
inline ExecutionResult result_aborted(double ms, uint64_t bytes) {
  ExecutionResult r = {ms, bytes, 1, 1};
  return r;
}
inline bool aborted(const volatile int* flag) {
  return flag != nullptr && *flag != 0;
}

}  // namespace

// Deterministic recompute workload. Each thread derives its output bytes from
// its index and a seed, mixed over `passes` rounds, so GPU time scales with
// size * passes and is tunable/comparable across runs.
__global__ void recompute_kernel(unsigned char* out, size_t n, int passes,
                                 unsigned int seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int x = (unsigned int)(i * 2654435761u) ^ seed;
  for (int p = 0; p < passes; ++p) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
  }
  out[i] = (unsigned char)(x >> 24);
}

extern "C" int cuda_device_count(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

extern "C" void cuda_snapshot(ResourceSnapshot* out) {
  std::memset(out, 0, sizeof(*out));
  out->gpu_util = -1.0f;
  out->gpu_mem_used = -1.0f;
  out->nvme_queue_depth = -1.0f;
  out->nvme_latency_us = -1.0f;
  out->pcie_bandwidth = -1.0f;

  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    out->gpu_mem_used = (float)(total_bytes - free_bytes);
  }

#ifdef GPUFLUX_NVML
  #include <nvml.h>
  nvmlInit();
  nvmlDevice_t dev;
  if (nvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS) {
    nvmlUtilization_t u;
    if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) {
      out->gpu_util = (float)u.gpu / 100.0f;
    }
  }
  nvmlShutdown();
#endif
}

extern "C" ExecutionResult cuda_move(ObjectDesc obj, CheckpointFn cb,
                                     volatile int* abort_flag) {
  auto t0 = std::chrono::steady_clock::now();

  unsigned char* pinned = nullptr;
  if (cudaMallocHost(&pinned, obj.size_bytes) != cudaSuccess) return result_fail();

  FILE* f = fopen(obj.nvme_path, "rb");
  if (f == nullptr) {
    cudaFreeHost(pinned);
    return result_fail();
  }
  size_t rd = fread(pinned, 1, obj.size_bytes, f);
  fclose(f);
  if (rd != obj.size_bytes) {
    cudaFreeHost(pinned);
    return result_fail();
  }

  if (cb) cb("move:host_read", elapsed_ms(t0), 0.5);
  if (aborted(abort_flag)) {
    cudaFreeHost(pinned);
    return result_aborted(elapsed_ms(t0), rd);
  }

  unsigned char* dptr = nullptr;
  if (cudaMalloc(&dptr, obj.size_bytes) != cudaSuccess) {
    cudaFreeHost(pinned);
    return result_fail();
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    cudaFreeHost(pinned);
    cudaFree(dptr);
    return result_fail();
  }
  cudaEvent_t ev_start = nullptr, ev_end = nullptr;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_end);

  if (aborted(abort_flag)) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    cudaFreeHost(pinned);
    cudaFree(dptr);
    return result_aborted(elapsed_ms(t0), rd);
  }

  cudaEventRecord(ev_start, stream);
  cudaMemcpyAsync(dptr, pinned, obj.size_bytes, cudaMemcpyHostToDevice, stream);
  cudaEventRecord(ev_end, stream);
  cudaStreamSynchronize(stream);

  float copy_ms = 0.0f;
  cudaEventElapsedTime(&copy_ms, ev_start, ev_end);

  if (cb) cb("move:h2d", elapsed_ms(t0), 1.0);
  if (aborted(abort_flag)) {
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    cudaFreeHost(pinned);
    cudaFree(dptr);
    return result_aborted(elapsed_ms(t0), obj.size_bytes);
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);
  cudaStreamDestroy(stream);
  cudaFreeHost(pinned);
  cudaFree(dptr);
  return result_ok(elapsed_ms(t0), obj.size_bytes);
}

extern "C" ExecutionResult cuda_recompute(ObjectDesc obj, CheckpointFn cb,
                                          volatile int* abort_flag) {
  auto t0 = std::chrono::steady_clock::now();

  unsigned char* dptr = nullptr;
  if (cudaMalloc(&dptr, obj.size_bytes) != cudaSuccess) return result_fail();
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    cudaFree(dptr);
    return result_fail();
  }

  if (cb) cb("recompute:start", 0.0, 0.0);
  if (aborted(abort_flag)) {
    cudaStreamDestroy(stream);
    cudaFree(dptr);
    return result_aborted(0.0, 0);
  }

  const int threads = 256;
  const int blocks =
      (int)((obj.size_bytes + threads - 1) / threads);
  recompute_kernel<<<blocks, threads, 0, stream>>>(dptr, obj.size_bytes,
                                                   obj.recompute_passes, 0x9e37u);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaStreamDestroy(stream);
    cudaFree(dptr);
    return result_fail();
  }

  if (cb) cb("recompute:done", elapsed_ms(t0), 1.0);
  if (aborted(abort_flag)) {
    cudaStreamDestroy(stream);
    cudaFree(dptr);
    return result_aborted(elapsed_ms(t0), obj.size_bytes);
  }

  cudaStreamDestroy(stream);
  cudaFree(dptr);
  return result_ok(elapsed_ms(t0), obj.size_bytes);
}
