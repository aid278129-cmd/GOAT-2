# ROUTE PREDICTION BENCHMARK REPORT

**Author:** Antigravity AI Engineering Team  
**Date:** September 28, 2026  
**System:** ANPR Traffic Intelligence Platform (Offline Prediction Benchmark Engine)  
**Execution:** `node scripts/benchmark_route_prediction.js`  
**Protocol:** Chronological 70% Train / 30% Test Split (Zero Data Leakage)  

---

## 1. BENCHMARK METHODOLOGY

To measure prediction accuracy with mathematical rigor, the benchmark implements:
1. **Chronological Splitting:** Detections are segmented into discrete journey sessions and sorted chronologically. The earliest 70% of journeys form the historical transition training model; the remaining 30% form the unseen evaluation test set.
2. **Zero Data Leakage:** Target transitions in the test period are strictly masked and never included in transition frequency tables.
3. **Multi-Horizon Metric Tracking:**
   - **Top-1 Accuracy:** Correct next camera is ranked #1.
   - **Top-2 Accuracy:** Correct next camera is within the first 2 candidates.
   - **Top-3 Accuracy:** Correct next camera is within the first 3 candidates.
   - **Mean Reciprocal Rank (MRR):** $\frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$.
   - **Coverage:** Percentage of prediction tasks where the model produces $\ge 1$ candidate.

---

## 2. REAL DATASET BENCHMARK RESULTS (`data/detections.json`)

- **Total Journeys:** 5 multi-camera journeys (Train: 3 journeys / 6 transitions, Test: 2 journeys / 6 transitions)
- **Status:** Evaluated on real Chennai traffic data (Corridor: CAM_01 $\rightarrow$ CAM_02 $\rightarrow$ CAM_03 $\rightarrow$ CAM_04)

| Model / Architecture | Top-1 Acc | Top-2 Acc | Top-3 Acc | MRR | Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 0: Global Most Common** | 33.33% | 50.00% | 83.33% | 0.528 | 100.00% |
| **Baseline 1: Most Common Outgoing** | 83.33% | 83.33% | 83.33% | 0.833 | 83.33% |
| **Model A: First-Order Markov** | 83.33% | 83.33% | 83.33% | 0.833 | 83.33% |
| **Model B: Second-Order Markov** | 83.33% | 83.33% | 83.33% | 0.833 | 83.33% |
| **Model C: Time-Aware Model** | 83.33% | 83.33% | 83.33% | 0.833 | 83.33% |
| **Model D: Direction-Aware Model** | 83.33% | 83.33% | 83.33% | 0.833 | 83.33% |
| **Model E: Full Contextual Predictor** | 83.33% | 83.33% | 83.33% | 0.833 | 100.00% |

*Note: In the real 37-detection dataset, test transitions follow the linear corridor with 1 test arrival at an unobserved node in the training split. Model E achieved 100% coverage via its hierarchical backoff.*

---

## 3. SIMULATION BENCHMARK RESULTS (12 CAMERAS / BRANCHING CORRIDORS)

- **Total Journeys:** 180 multi-camera journeys (Train: 125 journeys, Test: 55 journeys)
- **Total Prediction Events:** 194 unseen transition prediction tasks
- **Network Complexity:** 12 camera nodes with multiple branching choices, time-of-day peak flow biases, and 2nd-order path dependencies.

| Model / Architecture | Top-1 Acc | Top-2 Acc | Top-3 Acc | MRR | Coverage | Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 0: Global Most Common** | 11.86% | 37.11% | 68.04% | 0.396 | 100.00% | Minimal ($O(1)$) |
| **Baseline 1: Most Common Outgoing** | 85.57% | 100.00% | 100.00% | 0.928 | 100.00% | Low ($O(1)$) |
| **Model A: First-Order Markov** | 85.57% | 100.00% | 100.00% | 0.928 | 100.00% | Low ($O(K)$) |
| **Model B: Second-Order Markov** | 85.57% | 100.00% | 100.00% | 0.928 | 100.00% | Medium ($O(K)$) |
| **Model C: Time-Aware Model** | **86.08%** | **100.00%** | **100.00%** | **0.930** | 100.00% | Medium ($O(K)$) |
| **Model D: Direction-Aware Model** | 85.57% | 100.00% | 100.00% | 0.928 | 100.00% | Low ($O(K)$) |
| **Model E: Full Contextual Predictor** | 85.57% | 100.00% | 100.00% | 0.928 | 100.00% | Medium-High ($O(K)$) |

---

## 4. KEY BENCHMARK FINDINGS

1. **Top-2 and Top-3 Saturation:**
   - For all Markov, Outgoing, and Contextual models, **Top-2 and Top-3 accuracy is 100.00%**. The actual next camera was ALWAYS present within the top 2 candidates displayed in the Command Center UI.
2. **First-Order vs. Higher-Order Complexity:**
   - First-Order Markov ($P(\text{Next} \mid \text{Current})$) achieves **85.57% Top-1 Accuracy** across 194 test events in an urban branching network.
   - Time-Conditioned prediction ($P(\text{Next} \mid \text{Current}, \text{TimeBucket})$) provides a slight gain to **86.08% Top-1 Accuracy** by capturing morning vs evening rush-hour directional biases.
   - Second-order sequence conditioning performs on par with first-order when sequence history is sparse, validating the importance of automatic backoff.
3. **Model Selection Principle (Phase 27):**
   - Because First-Order Markov provides 85.57% Top-1, 100% Top-3, and an MRR of 0.928 with minimal computational overhead and instant explainability, it represents the ideal core statistical engine, enhanced with contextual time and direction penalties.
