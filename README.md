# GPUFlux

**Adaptive runtime for GPU data movement, recomputation, and remote execution**

GPUFlux decides whether to **move**, **recompute**, or **fetch** GPU data based on predicted
completion time, estimate uncertainty, resource contention, and dependency deadlines.

[![CI](https://github.com/swadhingoswami/GPUFlux/actions/workflows/ci.yml/badge.svg)](https://github.com/swadhingoswami/GPUFlux/actions/workflows/ci.yml)

---

## Why this exists (30-second version)

A GPU pipeline produces an intermediate object `X` that a later stage needs. When GPU
memory is constrained, `X` must be obtained one of several ways:

```text
Stage A
   |
   | produces X
   v
GPU memory full
   |
   +---- MOVE from storage
   +---- RECOMPUTE
   +---- REMOTE execution
   |
   v
Stage B needs X
```

The naive rule is:

```text
if move_time < recompute_time:
    move()
else:
    recompute()
```

That rule assumes costs are fixed. They are not. NVMe, PCIe, CPU, GPU, and network
contention change constantly, and so does the remaining time until stage B needs `X`.
A transfer that takes 20 ms when NVMe and PCIe are idle can take 50 ms under contention;
a recomputation can be delayed by a busy GPU. **The correct decision changes with the
system, and the runtime cannot know the future.**

## The key insight

GPUFlux does not try to predict one fixed cost per path. It tries to choose the path
**most likely to complete before the dependent computation needs the data** — that is,
it manages deadline risk under uncertainty.

```
   OBSERVE → HISTORY → PREDICT → ESTIMATE RISK → DECIDE → EXECUTE → MEASURE → LEARN
```

---

## Results (measured)

All numbers are from the development environment (Apple M3, 8 GB, Apple SSD, 256 MiB
object; mean ± std over 3–4 repeats). These are experimental measurements, not
production benchmarks.

**Deadline-aware beats expected-cost** (deadline D = 175 ms, 4 repeats)

```text
expected_cost   meet  64.2 ± 19.1 %
deadline_aware  meet  81.7 ± 12.3 %
```

**Replanning rescues a failing choice** (IO contention, 3 repeats)

```text
no fallback          meet  21.7 ± 20.2 %
with fallback        meet  93.3 ±  5.8 %   (mean wasted ≈ 22 ms)
```

**History beats a crude state-aware model** (IO contention, 3 repeats, mean regret)

```text
current-state model  regret 41.4 ± 7.8 ms
historical model     regret  5.6 ± 9.7 ms
```

**A third channel changes the landscape** (remote compute, 3 repeats, mean latency)

```text
local-only           693.7 ± 169.7 ms
with remote path     182.3 ±   0.4 ms
```

The same experiments show two negative results that matter: a naive state-aware model
can be **worse than a static baseline**, and the deadline inversion is **real but
hardware-sensitive**. Details in [Benchmark methodology](#benchmark-methodology).

## One decision, two states

The same object, the same pipeline — only the resource state changes.

```text
State: GPU busy (simulated load 0.7)          State: GPU idle
  move      181 ms  (predicted)                 move      181 ms
  recompute 252 ms  (predicted)                 recompute ~150 ms
  → GPUFlux chooses MOVE                        → GPUFlux chooses RECOMPUTE
```

With a remote path enabled (Phase 8), a fast remote node becomes the choice when
NVMe is contended and remote queue is low (`remote-load 0.1 → remote`), and the
runtime abandons it as remote load rises (`0.5, 0.9 → move`).

## Quick start

```sh
cargo test --release                                        # 23 unit tests
cargo run --release -p gpuflux-bench --bin demo -- --gpu-load=0.7
```

The demo runs the full decision loop against a simulated GPU load and prints, per
decision, the observed state, the predicted cost of each path, the chosen action, the
real measured cost, and whether the deadline was met:

```text
 dec    gpu    cpu   movePred   recPred   chosen  actual  met
   1   0.70   0.00      181       252       move     284    yes
   2   0.70   0.20      181       293       move     171    yes
   ...
 decisions : move=12 recompute=0    mean cost : 156.5 ms
```

At `--gpu-load=0`, recompute wins (~135 ms); at `--gpu-load=0.7`, move wins (~157 ms).
The decision engine does not change — only the observed state does.

---

## Architecture

```text
                       DecisionEngine (Rust)
                            │  policy.choose(ctx, predictions)
                     ┌──────▼────────┐
                     │   Predictor    │  CurrentState · Historical · OnlineRegression
                     └──────┬────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Current State + History   │  telemetry + redb observation store
              └─────────────┬──────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
  SimMoveExecutor     SimRecomputeExecutor   SimRemoteExecutor / CudaBackend
  (real SSD I/O)      (timed CPU fill)      (remote CPU + network / C++/CUDA)
```

The important design boundary: **a `Policy` emits an `Action`; an executor implements it.**
Prediction and decision logic never depend directly on CUDA or remote-execution details.
That is what keeps the decision engine backend-agnostic — the same code runs against the
SSD sim, the CPU fill, the simulated remote node, and the CUDA backend.

---

## Decision model

Intuition first: average latency is not enough. Two paths can have similar averages but
very different tails, and near a deadline the tail is what matters. GPUFlux scores each
candidate action `a` as

```
J(a) = E[T_a] + λ · P(T_a > D) + μ · U_a
a*   = argmin_a J(a)
```

| term | meaning | where it comes from |
|---|---|---|
| `E[T_a]` | expected completion time | EWMA history, or an online-learned model |
| `P(T_a > D)` | probability of missing the dependency deadline | log-normal fit through historical p50/p90 |
| `U_a` | uncertainty of the estimate (p90 − p50) | historical spread |
| `D` | remaining time until stage B needs X | dependency/deadline input |
| `λ, μ` | tunable risk weights | policy configuration (λ explored in the benchmark section) |

The decision loop, per decision:

```text
 Engine          Predictor          Store (redb)          Executor
   |  sample state  |                    |                    |
   |--------------->|                    |                    |
   |                | read history ----->|  aggregates per    |
   |                |                    |  (action,size,regime)|
   | predictions <--|  E, p50, p90, p95, |                    |
   |                |  P(T>D)            |                    |
   | policy.choose(ctx, pred)            |                    |
   | Action::Move --------------------------->                |
   |                |                    |   execute (chunked,
   |                |                    |   checkpoints, abort)
   | <---------------------------------------- actual, met, aborted
   | record(actual) |------------------->|  EWMA/var, reservoir |
   | predictor.update(action, state, actual)                  |
   | next decision  |                    |                    |
```

---

## Prediction and learning

The project evolved through five prediction levels, each motivated by a measurement:

```text
Level 0  Fixed            AlwaysMove / AlwaysRecompute
Level 1  Current state    analytic cost model from live telemetry
Level 2  History          EWMA mean/variance + p50/p90/p95 (history-as-primary)
Level 3  Uncertainty      distribution → P(T>D) (log-normal fit), U = p90 − p50
Level 4  Context-aware    online linear regression learns state → cost continuously
```

- **EWMA** adapts to drift: `mean_t = α·x_t + (1−α)·mean_{t−1}`.
- **Quantiles** come from a bounded reservoir of recent samples.
- **P(T>D)** is `1 − Φ((ln D − μ)/σ)` from a log-normal fit through p50/p90.
- **Online regression** (recursive least squares with forgetting) learns, for example,
  that recompute is insensitive to CPU pressure while move degrades with NVMe latency —
  no hand-tuned coefficients.

History lives in an embedded KV store (`redb`, not JSON): fast aggregates per
`(action, object-size, state-regime)` bucket (sample count, EWMA mean/variance, bounded
sample reservoir, deadline success rate) plus one `DecisionEvent` per decision
(predicted costs, chosen action, actual cost, deadline result, prediction error,
fallback, wasted time, regret). From Phase 4 onward, buckets are conditioned on a coarse
state *regime* (`cpu-lo/hi`, `io-lo/hi`, `gpu-lo/hi`) so history collected under heavy
contention does not pollute the low-contention estimate.

**A negative result worth stating:** the crude Level-1 model initially performed *worse*
than the static baseline. Having telemetry is not the same as having a correct model of
how resources affect execution. That failure is what motivated the history and
regression levels.

---

## What is real and what is simulated

| Component | Status |
|---|---|
| SSD movement | Real (`F_NOCACHE` / `F_FULLFSYNC` device I/O) |
| CPU recomputation | Real (deterministic CPU fill work) |
| CPU / IO contention | Real (spinning threads, device-reader threads) |
| GPU behavior | Simulated (GPU load is a parameter; NVML would supply it on real hardware) |
| CUDA backend (`gpuflux-cuda`) | Implemented, **not compiled/validated** (no nvcc/NVIDIA GPU here) |
| Remote execution | Simulated (modeled remote node + network, explicit parameters) |
| History store (`redb`) | Real |
| Online regression | Real, in-memory (not yet persisted) |

## What is (and is not) novel

No individual technique is novel: EWMA, online regression, log-normal risk modeling,
deadline scheduling, and checkpoint/replanning are all standard.

The contribution is the **integration and empirical study** of these mechanisms in one
runtime decision loop, and the measurements that come out of it — including the findings
that crude state-aware models can lose to static baselines, that history-as-primary
beats history-blended, that the deadline inversion is real but hardware-sensitive, and
that replanning recovers failed choices.

This is an experimental runtime, not a production scheduler, a storage system, or a
scheduling framework. Related but distinct work: GPUDirect Storage optimizes the data
path speed once a move is chosen; GPU memory managers use static eviction rules;
activation recomputation is decided at graph-build time; job schedulers allocate
resources rather than make per-object data-path decisions.

---

## Implementation phases

Each phase was added because the previous one's measurement exposed a gap — an
engineering/research feedback loop.

| Phase | What was added | Why | What the experiment showed |
|---|---|---|---|
| 0 | A→X→B benchmark, real SSD I/O, redb store | prove the problem exists | move CV 0.30 vs recompute CV 0.02 — move is genuinely variable |
| 1 | `Policy` trait, baselines, `DecisionEngine`, regret-vs-oracle | establish reference points | baselines set; move regret 49 ms, recompute 0.9 ms |
| 2 | telemetry sampler, current-state cost model, `ExpectedCost` | react to live state | naive state-aware policy **loses** to a static baseline (regret 6.7 vs 3.5 ms) |
| 3 | historical EWMA + quantiles (history-as-primary) | fix the over-reaction | prediction error roughly halved |
| 4 | contention injectors + regime-bucketed history | separate regimes | history regret 5.6±9.7 ms vs crude model 41.4±7.8 ms |
| 5 | deadline risk `P(T>D)` + `DeadlineAware` | near a deadline, tails matter | meet rate 81.7±12.3% vs 64.2±19.1% |
| 6 | replanning/fallback (checkpoints + abort) | predictions can be wrong mid-flight | meet 93.3±5.8% vs 21.7±20.2% |
| 7 | online regression (RLS + residual quantiles) | learn, don't hand-tune | discovers recompute is flat, move tracks NVMe latency |
| 8 | remote recompute path (3-way decision) | network + remote queue are real cost terms | local-only 694 ms → remote 182 ms |

## Benchmark methodology

- **Environment**: Apple M3 (8 GB, Apple SSD, macOS). One workload point: 256 MiB
  intermediate object, fill-pass recomputation.
- **Repetitions**: headline claims are 3–4 repeats, reported as mean ± std.
- **Contention**: induced CPU load (spinning cores) and IO load (F_NOCACHE reader
  threads); simulated GPU load via a parameter that inflates recompute work by
  `1/(1−gpu_load)` as real work, not a sleep.
- **Metrics**: primary metric is deadline-meet rate; mean cost regret vs an oracle
  (min over all measured paths) is reported alongside.
- **Real vs simulated**: see the table above. No CUDA number here is real — the CUDA
  backend has not been compiled or run.

---

## Repository layout

```text
crates/
  gpuflux/          core library
    src/decision/     policies + DecisionEngine (MOVE / RECOMPUTE / REMOTE)
    src/prediction/   cost model, historical (EWMA+quantiles), online regression, deadline risk
    src/executor/     executor traits + ExecutionControl (checkpoint/abort) + sim backends
    src/observation/  redb KV: fast aggregates + decision-event log
    src/telemetry/    SystemSampler (CPU/NVMe); CudaSampler on the GPU box
    src/contention/   CPU / IO / GPU contention injectors
  gpuflux-bench/     phase0–phase8 + demo binaries
  gpuflux-cuda/      C++/CUDA backend (standalone; requires nvcc)

.github/workflows/   CI (fmt/clippy/test/gate) + benchmark-report artifact
```

Other commands:

```sh
cargo run --release -p gpuflux-bench --bin phase5 -- --policy=deadline_aware --passes=2.5 --deadline-ms=175
cargo run --release -p gpuflux-bench --bin phase8 -- --policy=expected_cost --remote-load=0.1 --io-readers=3
```

Stores persist in the OS temp dir per policy; delete the `.db` files to reset history.

### CUDA backend

```sh
cd crates/gpuflux-cuda && cargo build --release    # requires nvcc + NVIDIA GPU
```

Implements the same executor/sampler traits over a plain C ABI (`cuda_backend.h`):
`recompute_kernel`, `cudaMemcpyAsync` move on a stream timed with events, NVML
telemetry. It is scaffolded but unvalidated until built on CUDA hardware.

---

## Limitations

- Measurements are from one workload point (256 MiB, fill-pass recompute) on Apple
  Silicon; GPU-box behavior (kernel contention, PCIe, NVML telemetry) is simulated here
  and unmeasured until CUDA hardware is available.
- The CUDA backend is written but never compiled or run in this repository (no nvcc).
- The remote node is simulated with explicit parameters, not a real second machine.
- The online-regression model is in-memory, not yet persisted to `redb`.
- The Phase 5 deadline inversion is real but hardware-sensitive; λ sensitivity was
  explored preliminarily, not as a full study.

## Roadmap

1. Compile + benchmark the CUDA backend on NVIDIA hardware (validate the FFI ABI)
2. Real (non-simulated) remote execution
3. Persist the online-regression model to `redb`
4. Broader benchmark matrix (sizes, workloads, contention profiles)
5. λ / μ parameter tuning or learning
6. Additional storage paths (Linux/O_DIRECT, etc.)

---

## Author

Built and maintained by **Swadhin Goswami** ([@swadhingoswami](https://github.com/swadhingoswami)).
Systems / runtime engineer (C++, Linux, storage) now focused on Rust + GPU systems,
runtime decision engines, and performance engineering. GPUFlux is a research / portfolio
project. *(Add LinkedIn / Twitter / email links here.)*

## Keywords

`#GPU #CUDA #NVIDIA #Rust #SystemsProgramming #HPC #GPUScheduling #DataMovement #Recomputation #OnlineLearning #PerformanceEngineering #OpenSource`
