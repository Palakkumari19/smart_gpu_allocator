# Real-Data Workload Trace Summary

> **AetherGrid Trace Builder**: Empirical multi-modal workload traces calibrated to 4x8GB GPU cluster target utilization presets (0.6, 0.9, 1.2).

## 1. Trace Generation Overview

Workload traces are generated directly from real production traces:
- **Arrival Timing**: Exact request timestamps from BurstGPT binned into simulation steps.
- **Multi-Modal Mix**: Blended mix combining BurstGPT LLM inference requests with UCI Video transcoding jobs.
- **Load Rescaling Formula**: Offered load is calibrated via queueing theory to reach target presets on the 4x8GB cluster:
  $$\rho = \frac{\lambda \cdot \bar{M} \cdot \bar{S}}{32.0 \text{ GB}}, \quad \text{load\_scale} = \frac{\rho_{\text{target}}}{\rho_{\text{raw}}}$$

---

## 2. Calibrated Trace Catalog

| Trace / Window | Target Load | Scale | Steps | Total Tasks | Video % | Real Fail % | Memory Mean (GB) | Dur Mean (Steps) | Peak Arrival/Step |
|---|---|---|---|---|---|---|---|---|---|
| `eval_median_hour_rho60` | `rho=0.545603` | 0.5456 | 360 | **144** | 23.61% | 3.47% | 5.36 | 14.02 | 3 |
| `eval_median_hour_rho90` | `rho=0.818404` | 0.8184 | 360 | **221** | 18.1% | 3.62% | 5.58 | 14.98 | 5 |
| `eval_median_hour_rho120` | `rho=1.091206` | 1.0912 | 360 | **289** | 17.99% | 2.42% | 5.38 | 14.73 | 5 |
| `eval_burstiest_window_rho60` | `rho=0.00849` | 0.0085 | 360 | **143** | 22.38% | 46.15% | 3.14 | 7.06 | 3 |
| `eval_burstiest_window_rho90` | `rho=0.012735` | 0.0127 | 360 | **204** | 14.22% | 50.49% | 2.89 | 6.80 | 4 |
| `eval_burstiest_window_rho120` | `rho=0.016979` | 0.0170 | 360 | **281** | 18.15% | 52.31% | 2.90 | 6.31 | 6 |

---

## 3. Detailed Trace Profiles

### Trace: `eval_median_hour_rho60` (Scale: 0.5456, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 233 / 64.72%).
- **Arrival Volatility**: Mean 0.4 tasks/step, Variance 0.351, Max 3 tasks/step.
- **Workload Composition**: 34 Video tasks (23.61%) + 110 LLM tasks (76.39%).
- **Real Fault Signal**: 5 tasks (3.47%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.07 GB, Mean **5.36 GB**, Median 5.97 GB, p95 7.57 GB, Max **7.93 GB**.
- **Duration**: Min 1 steps, Mean **14.02 steps**, Median 17 steps, p95 19 steps, Max **20 steps**.

### Trace: `eval_median_hour_rho90` (Scale: 0.8184, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 192 / 53.33%).
- **Arrival Volatility**: Mean 0.614 tasks/step, Variance 0.604, Max 5 tasks/step.
- **Workload Composition**: 40 Video tasks (18.1%) + 181 LLM tasks (81.9%).
- **Real Fault Signal**: 8 tasks (3.62%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.07 GB, Mean **5.58 GB**, Median 6.2 GB, p95 7.6 GB, Max **7.93 GB**.
- **Duration**: Min 1 steps, Mean **14.98 steps**, Median 18 steps, p95 19 steps, Max **20 steps**.

### Trace: `eval_median_hour_rho120` (Scale: 1.0912, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 173 / 48.06%).
- **Arrival Volatility**: Mean 0.803 tasks/step, Variance 0.931, Max 5 tasks/step.
- **Workload Composition**: 52 Video tasks (17.99%) + 237 LLM tasks (82.01%).
- **Real Fault Signal**: 7 tasks (2.42%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.07 GB, Mean **5.38 GB**, Median 5.95 GB, p95 7.49 GB, Max **7.95 GB**.
- **Duration**: Min 1 steps, Mean **14.73 steps**, Median 17 steps, p95 19 steps, Max **20 steps**.

### Trace: `eval_burstiest_window_rho60` (Scale: 0.0085, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 260 / 72.22%).
- **Arrival Volatility**: Mean 0.397 tasks/step, Variance 0.495, Max 3 tasks/step.
- **Workload Composition**: 32 Video tasks (22.38%) + 111 LLM tasks (77.62%).
- **Real Fault Signal**: 66 tasks (46.15%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.05 GB, Mean **3.14 GB**, Median 2.01 GB, p95 7.18 GB, Max **7.49 GB**.
- **Duration**: Min 1 steps, Mean **7.06 steps**, Median 3 steps, p95 17 steps, Max **19 steps**.

### Trace: `eval_burstiest_window_rho90` (Scale: 0.0127, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 249 / 69.17%).
- **Arrival Volatility**: Mean 0.567 tasks/step, Variance 0.962, Max 4 tasks/step.
- **Workload Composition**: 29 Video tasks (14.22%) + 175 LLM tasks (85.78%).
- **Real Fault Signal**: 103 tasks (50.49%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.05 GB, Mean **2.89 GB**, Median 1.07 GB, p95 6.54 GB, Max **7.95 GB**.
- **Duration**: Min 1 steps, Mean **6.8 steps**, Median 1 steps, p95 18 steps, Max **19 steps**.

### Trace: `eval_burstiest_window_rho120` (Scale: 0.0170, Seed: 42)

- **Step Resolution**: 10.0s/step across **360 steps** (Empty steps: 243 / 67.5%).
- **Arrival Volatility**: Mean 0.781 tasks/step, Variance 1.805, Max 6 tasks/step.
- **Workload Composition**: 51 Video tasks (18.15%) + 230 LLM tasks (81.85%).
- **Real Fault Signal**: 147 tasks (52.31%) originated from real BurstGPT requests aborted under server burst pressure.
- **VRAM Footprint**: Min 1.03 GB, Mean **2.9 GB**, Median 1.07 GB, p95 6.74 GB, Max **7.88 GB**.
- **Duration**: Min 1 steps, Mean **6.31 steps**, Median 1 steps, p95 18 steps, Max **20 steps**.

