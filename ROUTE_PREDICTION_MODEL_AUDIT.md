# ROUTE PREDICTION MODEL AUDIT

**Author:** Antigravity AI Engineering Team  
**Date:** September 28, 2026  
**System:** ANPR Traffic Intelligence Platform (Route Prediction Engine)  
**Status:** AUDIT COMPLETED  

---

## 1. WHERE TRANSITION COUNTS COME FROM

The transition frequency counts are calculated by [`buildTransitionModel(journeys)`](file:///d:/college%20work/Hackaton%20projects/backup-project-main/services/routePredictionService.js#L50-L94) in `services/routePredictionService.js`:
```javascript
function buildTransitionModel(journeys = []) {
  const singleTransitions = {};   // fromCam -> { toCam: count }
  const corridorTransitions = {}; // fromCam+midCam -> { toCam: count }
  const cameraTravelTimes = {};   // fromCam->toCam -> [travelTimeSeconds]
  ...
}
```
For each valid journey having $\ge 2$ sightings, the function iterates through chronological sighting pairs $(i-1, i)$ and increments:
- First-order: `singleTransitions[prevCam][currCam] = (count || 0) + 1`
- Second-order: `corridorTransitions[`${prevPrevCam}->${prevCam}`][currCam] = (count || 0) + 1`
- Empirical travel times: `cameraTravelTimes[`${prevCam}->${currCam}`].push(deltaSeconds)`

### Critical Audit Finding (The Cause of 38% / 37% / 25%):
In `services/trajectoryService.js` (line 616):
```javascript
const predictions = predictNextCamerasSync(activeJourney, journeys);
```
The caller passed `journeys`, which was **only the journey list belonging to the queried vehicle**, rather than the entire fleet-wide historical journey database.
When evaluating plate `MH47BP8265` (which had only 2 single-camera sightings separated by 38 hours, sessionized into two 1-sighting trips), `journeys` contained zero multi-camera transitions.
Consequently:
- `singleTransitions` was completely empty (`totalTransitions = 0`).
- The transition count for every candidate was 0 (`histCount = 0`).
- Because `totalTransitions === 0`, historical probability could not be computed, and the engine fell back to its heuristic prior and road distance scoring!

---

## 2. HOW PROBABILITIES ARE CALCULATED

When scoring candidate cameras from `currentCamId`, the engine computes a composite score:
$$\text{Score} = \text{Factor A (Historical Transition)} + \text{Factor B (Corridor Continuity)} + \text{Factor C (Road Topology)} + \text{Factor D (Direction Feasibility)}$$

1. **Factor A — 1st-Order Transition (Weight 0.45):**
   - If historical observations exist: $\text{score} += \left(\frac{\text{histCount}}{\text{totalTransitions}}\right) \times 0.45$.
   - If no historical observations exist: $\text{score} += 0.08$ (Baseline network prior).
2. **Factor B — 2nd-Order Continuity (Weight 0.25):**
   - If a prior camera `prevCamId` exists: $\text{score} += \left(\frac{\text{corrCount}}{\text{totalCorridorTransitions}}\right) \times 0.25$.
3. **Factor C — Road Topology Distance Bonus (Weight up to 0.20):**
   - Derived from OSRM road distance: $\text{distBonus} = \max\left(0.05, 0.20 - \frac{\text{distanceKm}}{20.0} \times 0.15\right)$.
4. **Factor D — Directional Alignment & Reversal Penalty:**
   - Immediate U-turn / reversal penalty: If candidate equals `prevCamId`, score is multiplied by $0.35$.
   - Outgoing road direction match: If candidate's allowed travel direction matches current camera's `possibleOutgoingRoadDirections`, $\text{score} += 0.10$.

### Exact Deconstruction of the 38% / 37% / 25% Prediction for CAM_02:
For `CAM_02` (Junction B) when no vehicle-specific multi-hop history was available:
- **Candidate CAM_01 (Junction A):**
  - Prior: $0.08$
  - Road distance: 2.75 km $\rightarrow \text{distBonus} = 0.20 - (2.75/20) \times 0.15 = 0.179 \approx 0.18$
  - Direction match (`northbound`): $+0.10$
  - **Raw Score:** $0.08 + 0.18 + 0.10 = 0.36$
- **Candidate CAM_03 (Junction C):**
  - Prior: $0.08$
  - Road distance: 3.62 km $\rightarrow \text{distBonus} = 0.20 - (3.62/20) \times 0.15 = 0.173 \approx 0.17$
  - Direction match (`southbound`): $+0.10$
  - **Raw Score:** $0.08 + 0.17 + 0.10 = 0.35$
- **Candidate CAM_04 (Highway Entry):**
  - Prior: $0.08$
  - Road distance: 5.62 km $\rightarrow \text{distBonus} = 0.20 - (5.62/20) \times 0.15 = 0.158 \approx 0.16$
  - Direction match (`westbound`): $+0.00$
  - **Raw Score:** $0.08 + 0.16 + 0.00 = 0.24$

**Score Normalization:**
- Total Score: $0.36 + 0.35 + 0.24 = 0.95$
- Normalized Probabilities:
  - $\text{CAM\_01} = \frac{0.36}{0.95} = 0.3789 \approx \mathbf{38\%}$
  - $\text{CAM\_03} = \frac{0.35}{0.95} = 0.3684 \approx \mathbf{37\%}$
  - $\text{CAM\_04} = \frac{0.24}{0.95} = 0.2526 \approx \mathbf{25\%}$

**Conclusion:** The values $38\% / 37\% / 25\%$ were **heuristic fallback scores** resulting from zero historical transition samples in the single vehicle scope, rather than statistical empirical transition frequencies.

---

## 3. WHETHER DEMO / SAMPLE DATA INFLUENCE THEM

- The raw detections file `data/detections.json` contains 37 detections across 17 vehicles, representing real and synthetic demonstration traffic on Chennai Central, T. Nagar, Perambur, and Guindy corridors.
- In the real dataset, all 5 multi-camera transitions originating from `CAM_02` transitioned to `CAM_03` ($100\%$). Zero real transitions went to `CAM_01` or `CAM_04`.
- Because the predictor was receiving only the plate's isolated sessions, it bypassed the real historical dataset and used the topology distance heuristic.

---

## 4. WHETHER TOPOLOGY DATA INFLUENCE THEM

- **Yes.** Topology route distances from `data/camera_network.json` directly provide the `distBonus` (Factor C: $0.05 - 0.20$).
- Connected camera lists (`connectedCameraIds`) and pre-cached OSRM road routes establish candidate eligibility.

---

## 5. WHETHER DIRECTION MODIFIES THEM

- **Yes.** Two directional mechanisms exist:
  1. Reversal penalty: Immediate return to the camera visited in the previous step ($candId == prevCamId$) is penalized by multiplying the score by $0.35$.
  2. Outgoing direction compatibility: Matches between camera outgoing directions and candidate arrival directions add $+0.10$.

---

## 6. WHETHER PROBABILITIES ARE NORMALIZED

- **Yes.** All candidate raw scores are summed:
  $$P(\text{cand}_i) = \frac{\text{rawScore}_i}{\sum_j \text{rawScore}_j}$$
  An explicit check guarantees that $\sum P = 1.00$ within floating point tolerance.

---

## 7. HOW ETA IS GENERATED

ETA combines OSRM road routing driving duration with empirical historical travel times:
- If historical travel times exist:
  $$\text{ETA Seconds} = \text{round}\left(0.60 \times \text{HistoricalMedianSeconds} + 0.40 \times \text{OSRMRouteDurationSeconds}\right)$$
- If no historical travel times exist:
  $$\text{ETA Seconds} = \text{OSRMRouteDurationSeconds}$$
Straight-line distance is **never** used for ETA.

---

## 8. HOW ACTIVE JOURNEY IS SELECTED

- `trajectoryService.segmentSightingsIntoJourneys()` splits detections chronologically whenever $\Delta t > 30\text{ min}$.
- `journeys[journeys.length - 1]` is designated as the `activeJourney`.
- Route prediction strictly evaluates `activeJourney.sessionSightings[last]`.

---

## 9. WHETHER FUTURE DATA CAN LEAK INTO PREDICTIONS

- In live runtime, future detections do not yet exist, so no leakage occurs.
- In offline benchmark testing, strict chronological splitting ($70\%$ train / $30\%$ test) is mandatory: transitions occurring in the evaluation window must never be included in the historical transition matrix.

---

## 10. WHETHER MANUALLY CONFIGURED VALUES INFLUENCE RESULTS

The following manual parameters influence the scoring balance:
- Historical weight: $0.45$
- Corridor continuity weight: $0.25$
- Baseline prior: $0.08$
- Topology distance bonus scale: $0.20 - (\text{dist}/20) \times 0.15$
- Reversal penalty: $0.35$
- Directional alignment bonus: $0.10$

These weights were manually tuned heuristics. In Phase 2, they are exposed transparently as distinct scoring components, and raw empirical transition frequencies are reported alongside adjusted scores.
