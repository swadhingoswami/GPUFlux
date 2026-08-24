#ifndef GPUFLUX_CUDA_BACKEND_H
#define GPUFLUX_CUDA_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI shared between the Rust decision core and the C++/CUDA execution layer.
// Kept deliberately small: the decision engine never touches CUDA, it only
// receives ExecutionResult.

typedef struct {
  uint64_t id;              // ObjectSpec.id
  uint64_t size_bytes;      // ObjectSpec.size_bytes
  int loc;                  // DataLoc: 0 gpu, 1 host, 2 nvme, 3 remote, 4 recompute
  int recompute_passes;     // work factor for the recompute kernel
  const char* nvme_path;    // persisted object file for MOVE (host-side NVMe)
} ObjectDesc;

typedef struct {
  double elapsed_ms;
  uint64_t bytes_moved;
  uint8_t success;          // 1 ok, 0 error
  uint8_t aborted;          // 1 stopped early (abort_flag set)
} ExecutionResult;

// Normalized telemetry snapshot (NVML + CUDA runtime). -1 means unavailable.
typedef struct {
  float gpu_util;           // fraction [0,1]
  float gpu_mem_used;       // bytes
  float nvme_queue_depth;   // (host-side, if measurable)
  float nvme_latency_us;    // (host-side, if measurable)
  float pcie_bandwidth;     // bytes/s
} ResourceSnapshot;

// Progress callback (matches Rust Progress). Called at coarse checkpoints.
typedef void (*CheckpointFn)(const char* phase, double elapsed_ms, double fraction_done);

int cuda_device_count(void);
void cuda_snapshot(ResourceSnapshot* out);

// MOVE: read persisted object from NVMe into pinned host memory, then
// cudaMemcpyAsync H2D. Returns total elapsed (host read + copy).
ExecutionResult cuda_move(ObjectDesc obj, CheckpointFn cb, volatile int* abort_flag);

// RECOMPUTE: launch the recompute kernel on the GPU. Returns kernel elapsed.
ExecutionResult cuda_recompute(ObjectDesc obj, CheckpointFn cb, volatile int* abort_flag);

#ifdef __cplusplus
}
#endif

#endif // GPUFLUX_CUDA_BACKEND_H
