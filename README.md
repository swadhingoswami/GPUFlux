<div align="center">

# GPUFlux

**Uncertainty-Aware Runtime for GPU Data Movement and Recomputation**

*"The best way to obtain GPU data changes with the system — GPUFlux learns when to move it, when to recompute it, and when to ship it elsewhere, under uncertainty."*

</div>

---

## 1. The story in three questions

1. **What is the problem?** The best way to obtain GPU data (move it, recompute it, or fetch it from a remote node) changes with system conditions. It is not a fixed choice.
2. **Why is it difficult?** The runtime observes the present, but it must decide for a future it cannot know — NVMe/PCIe/GPU/CPU contention and the exact next-use time are all uncertain.
3. **What are we doing?** GPUFlux observes current state, learns from historical execution data, estimates the future **cost and risk** of every candidate path, chooses the action most likely to complete before the dependent computation needs the data, and feeds the measured result back into its model.

```
   OBSERVE → HISTORY → PREDICT → ESTIMATE RISK → DECIDE → EXECUTE → MEASURE → LEARN
```

---

## 2. The problem

A GPU pipeline produces an intermediate object `X` that a later stage (stage B) needs. At decision time `X` may:

- already be in GPU memory,
- exist in host memory or on NVMe,
- be recomputable from scratch,
- or (Phase 8) be producible on a remote CPU.

When GPU memory is limited, the runtime has to choose **how to obtain `X`**. The naive solution is:

```text
if move_time < recompute_time:
    move()
else:
    recompute()
```

### 2.1 Why the naive solution fails — the bottleneck

The naive solution assumes costs are **fixed**. They are not:

```text
Today (t=0):               5 ms later (t=5ms):
  NVMe  → idle              NVMe  → congested
  GPU   → busy              GPU   → available
  CPU   → idle              CPU   → busy
  PCIe  → available         PCIe  → congested
```

The **bottleneck is not** *"how fast can we move 1 GB?"*
The bottleneck is *"which path is most likely to complete **before the dependent computation needs the data**?"*

A transfer that takes 20 ms when NVMe and PCIe are idle can take 50 ms when another workload uses them; a recomputation that normally takes 25 ms can be delayed by GPU contention. The runtime cannot know the future, so it must **manage uncertainty**, not pretend to eliminate it.

---

## 3. Why this isn't solved yet / who is already working on it

Each individual technique exists; the *integrated, measured* runtime does not.

| Related area | What it does | What it does NOT do |
|---|---|---|
| **GPUDirect Storage** (NVIDIA) | Speeds up the data path (CPU-bypass, `cudaMemcpy` from NVMe) | It does not *decide*; it optimizes the channel speed once a move is chosen |
| **GPU memory managers / caching** (GDS-integrated frameworks, memory pools) | Evict/retain policy for buffers | Static LRU/retention rules; no deadline-aware, learned move-vs-recompute choice |
| **Activation recomputation** (ML training) | Trade recompute vs store for activations | Decided once at graph-build time, not at runtime under contention |
| **GPU schedulers / orchestration** (Kubernetes device plugins, Slurm) | Allocate resources / place jobs | Do not decide per-intermediate-object data movement under uncertainty |
| **Learned schedulers / bandits / RL** | Online policies for scheduling | Typically job-level, not per-object with deadlines; rarely coupled to execution + replanning + measured feedback |
| **HPC data staging / prefetch** | Prefetch heuristics | No recompute-vs-move trade, no deadline risk, no online learning |

**Honest positioning.** No single component in GPUFlux is novel — EWMA, online regression, log-normal risk, and checkpoint-abort replanning are textbook. The contribution is:

- an **integrated runtime** in which all of these cooperate behind one small interface,
- a **measured** demonstration on real (contended) hardware that an uncertainty-aware runtime beats cost-only policies,
- and a set of **empirical findings** (crude state-aware models can lose to static baselines; history-as-primary beats history-blended; the deadline inversion is real but hardware-sensitive; replanning recovers failed choices).

---

## 4. How GPUFlux solves it

### 4.1 The decision formula

For each candidate action `a ∈ { MOVE, RECOMPUTE, REMOTE }`:

```
J(a) = E[T_a] + λ · P(T_a > D) + μ · U_a
a*   = argmin_a J(a)
```

| term | meaning | source |
|---|---|---|
| `E[T_a]` | expected completion time | EWMA history, or an online-learned model |
| `P(T_a > D)` | deadline-miss probability | log-normal fit through historical p50/p90 quantiles |
| `U_a` | estimate uncertainty (p90 − p50) | historical spread |
| `D` | remaining time until stage B needs X | dependency/deadline input |
| `λ, μ` | tunable risk weights | policy configuration (λ studied: see §7) |

### 4.2 The decision loop (sequence diagram)

```text
 Engine          Predictor          Store (redb)          Executor (backend)
   |                |                    |                     |
   | sample state   |                    |                     |
   |--------------->|                    |                     |
   |                | read history ----->|  aggregates per      |
   |                | (per regime bucket) |  (action,size,state) |
   | predictions <--|                    |                     |
   | (E,p50,p90,    |                    |                     |
   |  p95,P(T>D))   |                    |                     |
   |                |                    |                     |
   | policy.choose(ctx, pred)            |                     |
   | Action::Move --------------------------->                |
   |                |                    |     execute (chunked,
   |                |                    |     checkpoints,
   |                |                    |     abort flag)
   | <---------------------------------------- actual, met, aborted
   |                |                    |                     |
   | record(actual) |------------------->|  update EWMA/var,    |
   |                |                    |  reservoir, deadline |
   | predict.update(action, state, actual)                     |
   |                |  (online learner: RLS / history)         |
   | next decision  |                    |                     |
```

### 4.3 Architecture

```text
                         DecisionEngine (Rust)
                            │  policy.choose(ctx, predictions)
                     ┌──────▼────────┐
                     │   Predictor    │  CurrentState · Historical · OnlineRegression
                     └──────┬────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Current State + History   │  telemetry + redb ObservationStore
              └─────────────┬──────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
  SimMoveExecutor     SimRecomputeExecutor   SimRemoteExecutor / CudaBackend
  (real SSD I/O)      (timed CPU fill)      (remote CPU + network / C++CUDA)
```

**Design principle:** a `Policy` emits `Action`; the engine routes through an executor
trait; CUDA and remote execution never leak into prediction or decision.

### 4.4 The four execution backends

| backend | what it is | real / simulated |
|---|---|---|
| `SimMoveExecutor` | write X to SSD + read back, `F_NOCACHE`/`F_FULLFSYNC` | **real device I/O** |
| `SimRecomputeExecutor` | deterministic CPU fill, `passes` parameter | real work; GPU kernel substitute |
| `SimRemoteExecutor` | modeled remote CPU + network transfer (Phase 8) | simulated node (explicit params) |
| `CudaBackend` (`gpuflux-cuda`) | CUDA kernel + `cudaMemcpyAsync` + NVML | real CUDA on a GPU box |

---

## 5. How prediction works and how history is used

### 5.1 History data model (embedded KV, not JSON)

Two layers in `redb` (fast aggregates + raw events):

```text
AggregateRow  (per (action, object-size, state-regime) bucket)
├── sample_count
├── ewma_mean, ewma_variance          # online mean/var, first sample seeds
├── samples[]                         # bounded reservoir → p50/p90/p95
├── deadline_ok / deadline_total      # → deadline success rate
├── last_update, model_version

DecisionEvent (one per decision)
├── decision_id, object_id, timestamp, resource_snapshot
├── predicted_costs_ms (per channel)
├── chosen_action, actual_cost_ms
├── deadline_remaining_ms, deadline_met
├── prediction_error_ms, fallback_used, wasted_ms, regret_ms
```

**State-conditioned history (Phase 4).** Buckets are keyed by a coarse *regime*
suffix derived from the live state (`cpu-lo/hi`, `io-lo/hi`, `gpu-lo/hi`), so history
collected under heavy contention does not pollute the low-contention estimate.

### 5.2 Prediction levels (progressive)

```text
Level 0  Fixed          AlwaysMove / AlwaysRecompute
Level 1  Current state  analytic cost model from live telemetry
Level 2  History        EWMA mean/variance + p50/p90/p95 (history-as-primary)
Level 3  Uncertainty    distribution → P(T>D) (log-normal fit), U = p90−p50
Level 4  Context-aware  online linear regression learns state→cost continuously
```

- **EWMA** adapts when conditions change: `mean_t = α·x_t + (1−α)·mean_{t−1}`.
- **Quantiles** come from a bounded reservoir of recent samples.
- **P(T>D)** is `1 − Φ((ln D − μ)/σ)` from a log-normal fit through p50/p90.
- **Online regression** (recursive least squares with forgetting) learns, e.g.,
  that recompute is *insensitive* to CPU pressure while move degrades with NVMe
  latency — without any hand-tuned coefficients.

### 5.3 Deciding the "proper channel" — scenario table

| Live system state | move est | recompute est | remote est | GPUFlux picks | why |
|---|---|---|---|---|---|
| idle | 150 ms | 128 ms | 96 ms | **remote** (if present) | cheapest path |
| NVMe contended | 400 ms | 128 ms | 96 ms | **recompute / remote** | move is slow *and* risky |
| GPU busy | 150 ms | 280 ms | 120 ms | **move / remote** | recompute slowed by GPU |
| CPU busy | 150 ms | 190 ms | 96 ms | **remote / move** | recompute slowed by CPU |
| tight deadline | risky | safe | safe | **safe channel** | `P(T>D)` dominates |
| remote node loaded | 150 ms | 128 ms | 540 ms | **local** | remote queue too high |

The choice is *not* "pick the fastest average" — it is "pick the channel most likely
to finish before the deadline," i.e. **argmin of `E[T] + λ·P(T>D) + μ·U`**.

---

## 6. Implementation phases

| Phase | Built | Demonstrated result |
|---|---|---|
| 0 | A→X→B benchmark, real SSD I/O, redb store | move `CV 0.30` vs recompute `CV 0.02` — the problem is real |
| 1 | `Policy` trait, baseline policies, `DecisionEngine`, regret-vs-oracle, event log | baselines established |
| 2 | telemetry sampler, `CurrentStateCostModel`, `ExpectedCost` | crude state-aware policy **loses** to a static baseline |
| 3 | `HistoricalPredictor` (EWMA + quantiles, history-as-primary) | over-reaction fixed; prediction error halved |
| 4 | contention injectors (CPU/IO/GPU) + regime-bucketed history | `expected_cost` regret 41±8 ms vs `historical` 6±10 ms |
| 5 | `P(T>D)` + `DeadlineAware` (`J = E + λP + μU`) | deadline inversion: meet rate 81.7±12% vs 64.2±19% |
| 6 | replanning / fallback (progress checkpoints + abort) | failing policy rescued: 93.3±6% vs 21.7±20% meet |
| 7 | `OnlineRegressionPredictor` (RLS + residual quantiles) | learns true state→cost map; regret ~2× lower |
| 8 | remote recompute path (3-way decision) | local-only 694 ms → remote path 182 ms |

---

## 7. Benchmarks (evidence, step by step)

All numbers are real measurements on the dev machine (Apple M3, 8 GB, Apple SSD,
256 MiB object). Repeated-run stats use mean±std.

### Step 1 — Phase 0: prove the problem exists

```text
op         n    mean(ms) std(ms)  min    p50    p90    p95    max    CV
move        30    181      53    134    173    246    330    343   0.30
recompute   30    128       3    121    128    129    135    139   0.02
```

**Evidence:** move is genuinely variable (CV 0.30, max 2.7× median); recompute is
nearly deterministic. A fixed policy cannot be optimal for move.

### Step 2 — Phase 2: a naive state-aware policy can LOSE to static

Under CPU contention the crude model assumed "CPU busy ⇒ recompute slow" — wrong
(recompute here is memory-bound). `expected_cost` picked move and paid more:

```text
CPU burn=4:  expected_cost regret 6.7 ms  vs  always_recompute 3.5 ms
```

**Evidence:** reacting to state with bad assumptions is worse than not reacting.

### Step 3 — Phase 4: history fixes calibration (IO contention, 3× repeats)

```text
expected_cost   regret 41.4 ± 7.8 ms   (misled by io-induced cpu_util)
historical_cost regret  5.6 ± 9.7 ms
```

**Evidence:** the store's learned aggregates beat the hand-tuned analytic model.

### Step 4 — Phase 5: deadline-aware beats min-cost (4× repeats)

```text
expected_cost  meet 64.2 ± 19.1 %
deadline_aware meet 81.7 ± 12.3 %
λ sweep (D=175): λ=0 → 76%   λ=200 → 92% (mean 145 ms)   λ=1000 → 96%
```

**Evidence:** near a deadline, the *risky-but-cheaper* path loses to the *safer*
path — the exact GPUFlux thesis. Margin is real but hardware-sensitive.

### Step 5 — Phase 6: replanning rescues failed choices (3× repeats)

```text
always_move, no fallback    meet 21.7 ± 20.2 %
always_move, with fallback  meet 93.3 ±  5.8 %   (mean wasted ~22 ms)
```

**Evidence:** a path that is going to miss its deadline can be aborted and swapped,
converting a 0–40% meet policy into ~90%+.

### Step 6 — Phase 7: the model learns the workload's true response

```text
learned move      = [bias 180, cpu +4.6,  nvme_lat +54.6, queue 0] ms
learned recompute = [bias 147, cpu −12.3, nvme_lat −11.6, queue 0] ms
```

**Evidence:** recompute is flat (memory-bound); move degrades with NVMe latency.
The runtime discovered this from data — no coefficients hand-tuned.

### Step 7 — Phase 8: more channels change the decision landscape (3× repeats)

```text
remote OFF (local-only)   mean 693.7 ± 169.7 ms
remote ON  (fast remote)  mean 182.3 ±   0.4 ms
remote-load sweep: 0.1 → remote;  0.5 → move;  0.9 → move (remote queue too high)
```

**Evidence:** a third candidate path, priced with network + remote-queue terms,
is exploited when it pays and abandoned when the remote node is loaded.

### Step 8 — Demo (simulated GPU contention)

```sh
cargo run --release -p gpuflux-bench --bin demo -- --gpu-load=0.7
```

```text
 dec    gpu    cpu   movePred   recPred   chosen  actual  met
   1   0.70   0.00      181       252       move     284    yes
   2   0.70   0.20      181       293       move     171    yes
   ...
 decisions : move=12 recompute=0    mean cost : 156.5 ms
```

At `--gpu-load=0` recompute wins (~135 ms); at `0.7` move wins (~157 ms) — the
engine prices GPU contention with zero code changes.

---

## 8. Benefits

- **For GPU pipelines with intermediate data**: fewer deadline misses and lower
  tail latency when memory is constrained and the system is contended.
- **For operators**: no hand-tuning of "move vs recompute" thresholds — the
  runtime learns the workload's real contention sensitivity.
- **For the field**: a small, reproducible testbed to study uncertainty-aware
  runtime decisions (move / recompute / remote) end-to-end, with honest metrics.
- **For the author**: demonstrates systems engineering (Rust runtime, storage,
  telemetry, benchmarking), measurement methodology, and prediction/ML integration.

---

## 9. Repository layout & running

```text
crates/
  gpuflux/          core library
    src/decision/    policies + DecisionEngine (MOVE / RECOMPUTE / REMOTE)
    src/prediction/  cost model, historical (EWMA+quantiles), online regression, deadline risk
    src/executor/    traits + ExecutionControl (checkpoint/abort) + sim backends
    src/observation/ redb KV: fast aggregates + decision-event log
    src/telemetry/   SystemSampler (CPU/NVMe); CudaSampler on GPU box
    src/contention/  CPU / IO / GPU contention injectors
  gpuflux-bench/     phase0–phase8 + demo binaries
  gpuflux-cuda/      C++/CUDA backend (standalone; requires nvcc)

.github/workflows/   CI (fmt/clippy/test/gate) + benchmark-report artifact
```

```sh
cargo test --release                                  # 23 unit tests
cargo run --release -p gpuflux-bench --bin demo   -- --gpu-load=0.7
cargo run --release -p gpuflux-bench --bin phase5 -- --policy=deadline_aware --passes=2.5 --deadline-ms=175
cargo run --release -p gpuflux-bench --bin phase8 -- --policy=expected_cost --remote-load=0.1 --io-readers=3
```

Stores persist in the OS temp dir per policy; delete the `.db` files to reset history.

### CUDA backend

```sh
cd crates/gpuflux-cuda && cargo build --release   # requires nvcc + NVIDIA GPU
```

It implements the same executor/sampler traits over a plain C ABI
(`cuda_backend.h`): `recompute_kernel`, `cudaMemcpyAsync` move on a stream timed
with events, NVML telemetry.

---

## 10. Verification & honesty

- **Primary metric is deadline-meet rate**; cost regret is reported alongside.
- Repeated-run statistics (mean±std) are recorded for every headline claim (above).
- **No novelty is claimed** for any single technique; the contribution is the
  integrated, *measured* system and its empirical findings.

### Known limitations

- Dev numbers are for one workload point (256 MiB, fill-pass recompute) on Apple
  Silicon; the GPU-box behavior (kernel contention, PCIe, NVML telemetry) is
  simulated here and unmeasured until CUDA hardware is available.
- The `.cu` backend is written but **never compiled/run** in this repo (no nvcc).
- The remote node (Phase 8) is simulated with explicit parameters, not a real
  second machine.
- `OnlineRegressionPredictor` is in-memory (not yet persisted to redb).
- Phase 5's inversion margin is real but hardware-sensitive; λ sensitivity was
  explored preliminarily (§7).

### Roadmap

- [ ] Compile + benchmark the CUDA backend on a GPU box (validate the FFI ABI)
- [ ] Real (non-simulated) remote execution
- [ ] Persist the online-regression model to redb
- [ ] λ sensitivity study and parameter auto-tuning
- [ ] Linux/O_DIRECT cache-bypass for the move path

---

<div align="center">

**GPUFlux — Observe · Learn · Decide · Execute · Reassess · Improve**

</div>
