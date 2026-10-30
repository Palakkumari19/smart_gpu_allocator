# Empirical Trace Data Profile & Workload Characteristics

> **AetherGrid Real-Data Pipeline**: Analysis of empirical real-world video transcoding and LLM trace distributions for realistic GPU scheduling, failure modeling, and risk evaluation.

## 1. Executive Summary

This profiling report analyzes empirical distributions from two primary public production datasets:
1. **UCI Online Video Characteristics and Transcoding Time**: 68,784 transcoding jobs capturing multi-resolution duration, CPU memory (`umem`), and processing time (`utime`).
2. **BurstGPT Production Trace (`BurstGPT_1.csv`)**: 1,429,737 real-world LLM requests with request/response tokens, arrival timestamps, and explicit failure events under burst load.

---

## 2. Video Workload: UCI Transcoding Dataset

- **Dataset Source**: UCI Machine Learning Repository (Dataset 335 / CC BY 4.0).
- **Total Samples**: 68,784 rows.
- **Note on Measurement**: Transcoding metrics were gathered via FFmpeg on CPU nodes. Values serve as relative distribution baselines for memory pressure and task duration in our GPU scheduler.

### 2.1 Distribution Percentiles

| Metric | p5 | p25 | p50 (Median) | p75 | p95 | p99 |
|---|---|---|---|---|---|---|
| **Allocated Memory (`umem` in KB)** | 132,192 | 216,820 | 219,480 | 219,656 | 330,259 | 665,719 |
| **Processing Time (`utime` in sec)** | 0.72 | 2.10 | 4.41 | 10.43 | 39.35 | 81.91 |
| **Video Duration (sec)** | 42.2 | 106.8 | 239.1 | 379.3 | 750.4 | 1768.9 |

### 2.2 Resolution Breakdown

| Resolution | Job Count | Percentage |
|---|---|---|
| `480x360` | 17,916 | 26.05% |
| `320x240` | 13,649 | 19.84% |
| `176x144` | 12,012 | 17.46% |
| `1280x720` | 11,569 | 16.82% |
| `640x480` | 10,226 | 14.87% |
| `1920x1080` | 3,412 | 4.96% |

### 2.3 Codec Distribution

| Codec | Count |
|---|---|
| `h264` | 31,545 |
| `vp8` | 18,387 |
| `mpeg4` | 12,012 |
| `flv` | 6,840 |

---

## 3. LLM Workload: BurstGPT Trace Analysis

- **Dataset Source**: BurstGPT Production LLM Trace (`BurstGPT_1.csv` / CC BY 4.0).
- **Total Requests**: 1,429,737.
- **Failed Requests (`Response tokens == 0`)**: 25,443 (1.780%).
- **Failure Signal**: Zero response tokens indicate requests terminated by the serving engine under severe memory pressure, request queue overflow, or timeouts.

### 3.1 Token Distribution Percentiles

| Metric | p5 | p25 | p50 (Median) | p75 | p95 | p99 |
|---|---|---|---|---|---|---|
| **Request (Prompt) Tokens** | 60 | 137 | 251 | 758 | 2,461 | 3,387 |
| **Response Tokens (All)** | 2 | 7 | 34 | 134 | 460 | 1,491 |
| **Response Tokens (Successful Only)** | 2 | 7 | 36 | 136 | 466 | 1,505 |

### 3.2 Temporal Arrival & Hourly Burstiness

- **Active Monitoring Window**: 1,428 hours (~60 calendar days).
- **Hourly Arrival Rate**: Mean of **1001.2** req/hr (Min: 1, Max: **33,344** req/hr).

| Arrival Metric | p5 | p25 | p50 (Median) | p75 | p95 | p99 |
|---|---|---|---|---|---|---|
| **Requests per Hour** | 4 | 51 | 217 | 982 | 4,144 | 11,541 |

### 3.3 Significant Failure Spikes under Burst Pressure

The trace exhibits localized burst hours where failure rate spikes dramatically due to cluster overload:

| Hour Offset | Total Requests | Failed Requests | Failure Rate (%) |
|---|---|---|---|
| Hour `1236` | 16,902 | 10,493 | **62.08%** |
| Hour `693` | 165 | 83 | **50.30%** |
| Hour `715` | 1,646 | 763 | **46.35%** |
| Hour `716` | 2,160 | 969 | **44.86%** |
| Hour `717` | 703 | 303 | **43.10%** |

---

## 4. Evaluation of Complementary Production Traces

### 4.1 Azure LLM Inference Traces (2023 & 2024)
- **Characteristics**: Production traces from Microsoft Azure OpenAI service capturing prompt context lengths, generation lengths, and inter-arrival timestamps.
- **Relevance**: Directly corroborates BurstGPT's heavy-tailed token distribution and provides an independent benchmark for cross-validating multi-modal LLM request dynamics.

### 4.2 Alibaba GPU Cluster Trace (`cluster-trace-gpu-v2020`)
- **Characteristics**: Large-scale production trace encompassing thousands of heterogeneous GPUs over several months.
- **Relevance**: Provides ground-truth distributions for job queue wait times, multi-GPU allocation sizing, and hardware preemption/status flags.
- **Sampling Strategy**: Due to multi-gigabyte uncompressed volume, subsampling representative 24-hour job arrival slices ensures rapid replay while maintaining realistic inter-arrival burstiness.

### 4.3 Hugging Face Gaming GPU Dataset (`gaming_gpu_dataset_3000`)
- **Critique & Exclusion**: Examination reveals synthetic, uniformly distributed feature curves (GPU model, game title, resolution, temperature, power, VRAM) rather than genuine sensor telemetry.
- **Domain Mismatch**: Real-time graphics workloads continuously hold fixed display buffers and do not follow the queuing, preemption, or batch completion dynamics of compute job schedulers.
- **Decision**: **Skipped** for scheduling evaluations. We rely strictly on real telemetry from UCI transcoding and BurstGPT.

---

## 5. Integration with AetherGrid Risk Engine

The empirical findings directly inform realistic task generation in [`realdata/`](file:///home/bmsit/Downloads/gpu-draft-main/realdata):
1. **Memory & Duration Distributions**: UCI `umem` and `utime` percentiles map directly into `EnhancedTask.memory_required` (1–8 GB) and `EnhancedTask.execution_time` (simulation steps).
2. **Burst Arrival Patterns**: BurstGPT hourly percentiles provide real-world arrival rate multipliers across off-peak, steady-state, and extreme burst regimes.
3. **Empirical Risk Signals**: Real failure spikes (such as Hour 1236 with 62% failure rate) enable calibrated fault overlays via `gpu.inject_fault()` to test how effectively Risk-Aware MARL protects against cascading cluster outages compared to traditional baselines.
