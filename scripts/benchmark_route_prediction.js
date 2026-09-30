'use strict';

/**
 * Offline Route Prediction Benchmark Engine
 * Evaluates Next-Camera Prediction Models and ETA Accuracy
 * Using Chronological Train/Test Split (Zero Data Leakage)
 */

const fs = require('fs');
const path = require('path');
const { segmentSightingsIntoJourneys, getPlateSightings, normalizePlate } = require('../services/trajectoryService');
const { haversineDistance } = require('../services/geoService');

function formatCamId(id) {
  if (!id) return '';
  const num = parseInt(String(id).replace(/[^0-9]/g, ''), 10);
  if (isNaN(num)) return String(id);
  return `CAM_${String(num).padStart(2, '0')}`;
}

function getTimeBucket(isoTimestamp) {
  const d = new Date(isoTimestamp);
  const hour = d.getUTCHours();
  if (hour >= 6 && hour < 11) return 'MORNING_PEAK';
  if (hour >= 11 && hour < 16) return 'MIDDAY';
  if (hour >= 16 && hour < 21) return 'EVENING_PEAK';
  return 'NIGHT';
}

/**
 * Extracts chronological journeys and splits them chronologically into Train (70%) and Test (30%).
 */
function prepareDataset(detectionsPath, config = { trajectorySessionGapMinutes: 30 }) {
  const rawDetections = JSON.parse(fs.readFileSync(detectionsPath, 'utf8'));

  // Group by plate and segment into journeys
  const plates = [...new Set(rawDetections.map((d) => normalizePlate(d.plate)))];
  const allJourneys = [];

  plates.forEach((plate) => {
    const sightings = getPlateSightings(plate, rawDetections);
    const journeys = segmentSightingsIntoJourneys(sightings, config);
    journeys.forEach((j) => {
      const s = j.sessionSightings || j.sightings || [];
      if (s.length >= 2) {
        allJourneys.push(j);
      }
    });
  });

  // Sort journeys by their start time chronologically
  allJourneys.sort((a, b) => new Date(a.sessionStart).getTime() - new Date(b.sessionStart).getTime());

  // 70% Train / 30% Test split by time
  const splitIndex = Math.max(1, Math.floor(allJourneys.length * 0.70));
  const trainJourneys = allJourneys.slice(0, splitIndex);
  const testJourneys = allJourneys.slice(splitIndex);

  // Generate test events (next-camera prediction tasks)
  const testEvents = [];
  testJourneys.forEach((journey) => {
    const sightings = journey.sessionSightings || journey.sightings || [];
    for (let i = 1; i < sightings.length; i++) {
      const prevSightings = sightings.slice(0, i);
      const targetSighting = sightings[i];
      testEvents.push({
        journeyId: journey.journeyId,
        plate: journey.plate,
        historySightings: prevSightings,
        currentCamera: formatCamId(prevSightings[prevSightings.length - 1].cameraId),
        previousCamera: prevSightings.length >= 2 ? formatCamId(prevSightings[prevSightings.length - 2].cameraId) : null,
        currentTimestamp: prevSightings[prevSightings.length - 1].timestamp,
        currentDirection: prevSightings[prevSightings.length - 1].direction || 'unknown',
        targetCamera: formatCamId(targetSighting.cameraId),
        targetTimestamp: targetSighting.timestamp,
        actualTravelTimeSeconds: Math.round((new Date(targetSighting.timestamp).getTime() - new Date(prevSightings[prevSightings.length - 1].timestamp).getTime()) / 1000),
      });
    }
  });

  return { allJourneys, trainJourneys, testJourneys, testEvents, rawDetectionsCount: rawDetections.length };
}

/**
 * Builds training models from historical trainJourneys only.
 */
function trainModels(trainJourneys, cameraTopology = null) {
  const globalDestFreq = {};
  const firstOrderTransitions = {}; // from -> { to: count }
  const secondOrderTransitions = {}; // `${prev}->${curr}` -> { to: count }
  const timeConditionedTransitions = {}; // `${curr}:${bucket}` -> { to: count }
  const travelTimes = {}; // `${from}->${to}` -> [seconds]

  trainJourneys.forEach((journey) => {
    const sightings = journey.sessionSightings || journey.sightings || [];
    for (let i = 1; i < sightings.length; i++) {
      const fromCam = formatCamId(sightings[i - 1].cameraId);
      const toCam = formatCamId(sightings[i].cameraId);
      if (!fromCam || !toCam || fromCam === toCam) continue;

      // Global destination frequency
      globalDestFreq[toCam] = (globalDestFreq[toCam] || 0) + 1;

      // 1st order
      if (!firstOrderTransitions[fromCam]) firstOrderTransitions[fromCam] = {};
      firstOrderTransitions[fromCam][toCam] = (firstOrderTransitions[fromCam][toCam] || 0) + 1;

      // Travel time
      const t1 = new Date(sightings[i - 1].timestamp).getTime();
      const t2 = new Date(sightings[i].timestamp).getTime();
      if (!isNaN(t1) && !isNaN(t2) && t2 > t1) {
        const sec = Math.round((t2 - t1) / 1000);
        if (sec > 0 && sec < 10800) {
          const key = `${fromCam}->${toCam}`;
          if (!travelTimes[key]) travelTimes[key] = [];
          travelTimes[key].push(sec);
        }
      }

      // Time bucket
      const bucket = getTimeBucket(sightings[i - 1].timestamp);
      const timeKey = `${fromCam}:${bucket}`;
      if (!timeConditionedTransitions[timeKey]) timeConditionedTransitions[timeKey] = {};
      timeConditionedTransitions[timeKey][toCam] = (timeConditionedTransitions[timeKey][toCam] || 0) + 1;

      // 2nd order
      if (i >= 2) {
        const prevCam = formatCamId(sightings[i - 2].cameraId);
        if (prevCam && prevCam !== fromCam) {
          const corridorKey = `${prevCam}->${fromCam}`;
          if (!secondOrderTransitions[corridorKey]) secondOrderTransitions[corridorKey] = {};
          secondOrderTransitions[corridorKey][toCam] = (secondOrderTransitions[corridorKey][toCam] || 0) + 1;
        }
      }
    }
  });

  // Calculate median travel times
  const medianTravelTimes = {};
  Object.entries(travelTimes).forEach(([pair, arr]) => {
    arr.sort((a, b) => a - b);
    const mid = Math.floor(arr.length / 2);
    medianTravelTimes[pair] = arr.length % 2 !== 0 ? arr[mid] : Math.round((arr[mid - 1] + arr[mid]) / 2);
  });

  return {
    globalDestFreq,
    firstOrderTransitions,
    secondOrderTransitions,
    timeConditionedTransitions,
    travelTimes,
    medianTravelTimes,
    cameraTopology,
  };
}

/**
 * Predictor implementations
 */
const Predictors = {
  // Baseline 0: Global Most Common Next Camera
  baseline0: (event, model) => {
    const entries = Object.entries(model.globalDestFreq);
    if (entries.length === 0) return [];
    entries.sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((s, e) => s + e[1], 0);
    return entries.map(([cam, cnt]) => ({
      cameraId: cam,
      probability: Number((cnt / total).toFixed(4)),
      confidence: 'LOW',
      support: cnt,
    }));
  },

  // Baseline 1: Most Common Outgoing Camera
  baseline1: (event, model) => {
    const outgoing = model.firstOrderTransitions[event.currentCamera] || {};
    const entries = Object.entries(outgoing);
    if (entries.length === 0) return [];
    entries.sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((s, e) => s + e[1], 0);
    return entries.map(([cam, cnt]) => ({
      cameraId: cam,
      probability: Number((cnt / total).toFixed(4)),
      confidence: cnt >= 5 ? 'HIGH' : cnt >= 2 ? 'MEDIUM' : 'LOW',
      support: cnt,
    }));
  },

  // Model A: First-Order Markov Model
  modelA_firstOrder: (event, model) => {
    const outgoing = model.firstOrderTransitions[event.currentCamera] || {};
    const entries = Object.entries(outgoing);
    if (entries.length === 0) return [];
    entries.sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((s, e) => s + e[1], 0);
    return entries.map(([cam, cnt]) => ({
      cameraId: cam,
      probability: Number((cnt / total).toFixed(4)),
      confidence: cnt >= 10 ? 'HIGH' : cnt >= 3 ? 'MEDIUM' : 'LOW',
      support: cnt,
      method: 'FIRST_ORDER',
    }));
  },

  // Model B: Second-Order Markov Model (with First-Order Backoff)
  modelB_secondOrder: (event, model) => {
    if (event.previousCamera) {
      const corridorKey = `${event.previousCamera}->${event.currentCamera}`;
      const secondOrder = model.secondOrderTransitions[corridorKey] || {};
      const entries = Object.entries(secondOrder);
      const total = entries.reduce((s, e) => s + e[1], 0);
      if (total >= 2) {
        entries.sort((a, b) => b[1] - a[1]);
        return entries.map(([cam, cnt]) => ({
          cameraId: cam,
          probability: Number((cnt / total).toFixed(4)),
          confidence: cnt >= 8 ? 'HIGH' : cnt >= 3 ? 'MEDIUM' : 'LOW',
          support: cnt,
          method: 'SECOND_ORDER',
        }));
      }
    }
    // Backoff to First-Order
    return Predictors.modelA_firstOrder(event, model).map((c) => ({ ...c, method: 'FIRST_ORDER_BACKOFF' }));
  },

  // Model C: Time-Conditioned Predictor
  modelC_timeAware: (event, model) => {
    const bucket = getTimeBucket(event.currentTimestamp);
    const key = `${event.currentCamera}:${bucket}`;
    const timeTrans = model.timeConditionedTransitions[key] || {};
    const entries = Object.entries(timeTrans);
    const total = entries.reduce((s, e) => s + e[1], 0);
    if (total >= 2) {
      entries.sort((a, b) => b[1] - a[1]);
      return entries.map(([cam, cnt]) => ({
        cameraId: cam,
        probability: Number((cnt / total).toFixed(4)),
        confidence: cnt >= 8 ? 'HIGH' : cnt >= 2 ? 'MEDIUM' : 'LOW',
        support: cnt,
        method: 'TIME_CONDITIONED',
      }));
    }
    return Predictors.modelA_firstOrder(event, model).map((c) => ({ ...c, method: 'FIRST_ORDER_BACKOFF' }));
  },

  // Model D: Direction-Aware Predictor
  modelD_directionAware: (event, model) => {
    const firstOrder = Predictors.modelA_firstOrder(event, model);
    if (firstOrder.length === 0) return [];

    let totalScore = 0;
    const scored = firstOrder.map((c) => {
      let score = c.probability;
      // Penalize immediate return / U-turn
      if (event.previousCamera && c.cameraId === event.previousCamera) {
        score *= 0.35;
      }
      totalScore += score;
      return { ...c, rawScore: score, method: 'DIRECTION_AWARE' };
    });

    return scored.map((c) => ({
      ...c,
      probability: Number((c.rawScore / totalScore).toFixed(4)),
    }));
  },

  // Model E: Full Contextual Predictor with Hierarchy
  modelE_fullContextual: (event, model) => {
    // 1. Check Second-Order with Context
    if (event.previousCamera) {
      const corridorKey = `${event.previousCamera}->${event.currentCamera}`;
      const secondOrder = model.secondOrderTransitions[corridorKey] || {};
      const entries = Object.entries(secondOrder);
      const total = entries.reduce((s, e) => s + e[1], 0);

      if (total >= 3) {
        entries.sort((a, b) => b[1] - a[1]);
        const scored = entries.map(([cam, cnt]) => {
          let score = cnt / total;
          if (cam === event.previousCamera) score *= 0.35;
          return { cameraId: cam, score, count: cnt };
        });
        const sumScore = scored.reduce((s, x) => s + x.score, 0) || 1;
        return scored.map((x) => ({
          cameraId: x.cameraId,
          probability: Number((x.score / sumScore).toFixed(4)),
          confidence: x.count >= 8 ? 'HIGH' : 'MEDIUM',
          support: x.count,
          method: 'SECOND_ORDER_CONTEXTUAL',
        }));
      }
    }

    // 2. Check Time-Conditioned
    const bucket = getTimeBucket(event.currentTimestamp);
    const timeKey = `${event.currentCamera}:${bucket}`;
    const timeTrans = model.timeConditionedTransitions[timeKey] || {};
    const timeEntries = Object.entries(timeTrans);
    const timeTotal = timeEntries.reduce((s, e) => s + e[1], 0);

    if (timeTotal >= 3) {
      timeEntries.sort((a, b) => b[1] - a[1]);
      return timeEntries.map(([cam, cnt]) => ({
        cameraId: cam,
        probability: Number((cnt / timeTotal).toFixed(4)),
        confidence: cnt >= 5 ? 'MEDIUM' : 'LOW',
        support: cnt,
        method: 'TIME_CONDITIONED_BACKOFF',
      }));
    }

    // 3. First-Order with Direction Penalty
    const firstOrder = Predictors.modelA_firstOrder(event, model);
    if (firstOrder.length > 0) {
      let sumScore = 0;
      const scored = firstOrder.map((c) => {
        let score = c.probability;
        if (event.previousCamera && c.cameraId === event.previousCamera) score *= 0.35;
        sumScore += score;
        return { ...c, score };
      });
      return scored.map((c) => ({
        cameraId: c.cameraId,
        probability: Number((c.score / sumScore).toFixed(4)),
        confidence: c.support >= 5 ? 'MEDIUM' : 'LOW',
        support: c.support,
        method: 'FIRST_ORDER_BACKOFF',
      }));
    }

    // 4. Baseline Prior
    return Predictors.baseline0(event, model).map((c) => ({ ...c, confidence: 'INSUFFICIENT_DATA', method: 'NETWORK_PRIOR' }));
  },
};

/**
 * Runs evaluation metrics across test events.
 */
function evaluatePredictor(name, predictorFn, testEvents, model) {
  let evaluatedCount = 0;
  let abstentions = 0;
  let top1Correct = 0;
  let top2Correct = 0;
  let top3Correct = 0;
  let reciprocalRankSum = 0;

  const confAccuracy = {
    HIGH: { correct: 0, total: 0 },
    MEDIUM: { correct: 0, total: 0 },
    LOW: { correct: 0, total: 0 },
    INSUFFICIENT_DATA: { correct: 0, total: 0 },
  };

  const calibrationBuckets = [
    { label: '0-20%', min: 0.0, max: 0.2, probs: [], correct: 0, total: 0 },
    { label: '20-40%', min: 0.2, max: 0.4, probs: [], correct: 0, total: 0 },
    { label: '40-60%', min: 0.4, max: 0.6, probs: [], correct: 0, total: 0 },
    { label: '60-80%', min: 0.6, max: 0.8, probs: [], correct: 0, total: 0 },
    { label: '80-100%', min: 0.8, max: 1.0, probs: [], correct: 0, total: 0 },
  ];

  testEvents.forEach((event) => {
    evaluatedCount++;
    const preds = predictorFn(event, model);
    if (!preds || preds.length === 0) {
      abstentions++;
      return;
    }

    const rank = preds.findIndex((p) => p.cameraId === event.targetCamera) + 1;
    if (rank === 1) top1Correct++;
    if (rank > 0 && rank <= 2) top2Correct++;
    if (rank > 0 && rank <= 3) top3Correct++;

    if (rank > 0) {
      reciprocalRankSum += 1.0 / rank;
    }

    // Top-1 prediction confidence and calibration evaluation
    const topPred = preds[0];
    const isTopCorrect = rank === 1;
    const conf = topPred.confidence || 'LOW';
    if (confAccuracy[conf]) {
      confAccuracy[conf].total++;
      if (isTopCorrect) confAccuracy[conf].correct++;
    }

    // Calibration bucket for top prediction
    const pVal = topPred.probability;
    const bucket = calibrationBuckets.find((b) => pVal >= b.min && (pVal < b.max || (b.max === 1.0 && pVal <= 1.0)));
    if (bucket) {
      bucket.probs.push(pVal);
      bucket.total++;
      if (isTopCorrect) bucket.correct++;
    }
  });

  const activeCount = evaluatedCount - abstentions;
  const coverage = evaluatedCount > 0 ? (activeCount / evaluatedCount) * 100 : 0;
  const top1Acc = evaluatedCount > 0 ? (top1Correct / evaluatedCount) * 100 : 0;
  const top2Acc = evaluatedCount > 0 ? (top2Correct / evaluatedCount) * 100 : 0;
  const top3Acc = evaluatedCount > 0 ? (top3Correct / evaluatedCount) * 100 : 0;
  const mrr = evaluatedCount > 0 ? reciprocalRankSum / evaluatedCount : 0;

  return {
    name,
    totalEvents: evaluatedCount,
    abstentions,
    coverage: Number(coverage.toFixed(2)),
    top1: Number(top1Acc.toFixed(2)),
    top2: Number(top2Acc.toFixed(2)),
    top3: Number(top3Acc.toFixed(2)),
    mrr: Number(mrr.toFixed(3)),
    confAccuracy: {
      HIGH: confAccuracy.HIGH.total > 0 ? Number(((confAccuracy.HIGH.correct / confAccuracy.HIGH.total) * 100).toFixed(2)) : null,
      MEDIUM: confAccuracy.MEDIUM.total > 0 ? Number(((confAccuracy.MEDIUM.correct / confAccuracy.MEDIUM.total) * 100).toFixed(2)) : null,
      LOW: confAccuracy.LOW.total > 0 ? Number(((confAccuracy.LOW.correct / confAccuracy.LOW.total) * 100).toFixed(2)) : null,
    },
    calibration: calibrationBuckets.map((b) => ({
      label: b.label,
      total: b.total,
      meanProb: b.probs.length > 0 ? Number((b.probs.reduce((s, p) => s + p, 0) / b.probs.length).toFixed(4)) : 0,
      actualAcc: b.total > 0 ? Number(((b.correct / b.total) * 100).toFixed(2)) : 0,
    })),
  };
}

/**
 * ETA Benchmark for correctly predicted transitions.
 */
function evaluateETA(testEvents, model, topology) {
  const osrmErrors = [];
  const medianErrors = [];
  const contextualErrors = [];

  testEvents.forEach((event) => {
    const actualSec = event.actualTravelTimeSeconds;
    if (actualSec <= 0 || actualSec > 7200) return;

    // 1. OSRM ETA from topology
    const key = `${event.currentCamera}:${event.targetCamera}`;
    const topoRoute = topology && topology.routes ? topology.routes[key] : null;
    const osrmSec = topoRoute ? topoRoute.durationSeconds : null;

    if (osrmSec !== null) {
      osrmErrors.push(Math.abs(osrmSec - actualSec));
    }

    // 2. Historical median travel time
    const histKey = `${event.currentCamera}->${event.targetCamera}`;
    const histMedian = model.medianTravelTimes[histKey];
    if (histMedian !== undefined) {
      medianErrors.push(Math.abs(histMedian - actualSec));
    }

    // 3. Contextual ETA (60% historical median + 40% OSRM)
    if (osrmSec !== null && histMedian !== undefined) {
      const contextualSec = Math.round(histMedian * 0.6 + osrmSec * 0.4);
      contextualErrors.push(Math.abs(contextualSec - actualSec));
    } else if (histMedian !== undefined) {
      contextualErrors.push(Math.abs(histMedian - actualSec));
    } else if (osrmSec !== null) {
      contextualErrors.push(Math.abs(osrmSec - actualSec));
    }
  });

  function stats(arr) {
    if (arr.length === 0) return { mae: null, median: null, p90: null, count: 0 };
    arr.sort((a, b) => a - b);
    const sum = arr.reduce((s, v) => s + v, 0);
    const mid = Math.floor(arr.length / 2);
    const p90Idx = Math.floor(arr.length * 0.9);
    return {
      mae: Number((sum / arr.length).toFixed(1)),
      median: arr[mid],
      p90: arr[p90Idx],
      count: arr.length,
    };
  }

  return {
    osrm: stats(osrmErrors),
    historicalMedian: stats(medianErrors),
    contextual: stats(contextualErrors),
  };
}

// ─────────────────────────────────────────────
// MAIN EXECUTION
// ─────────────────────────────────────────────
function runAllBenchmarks() {
  console.log('======================================================================');
  console.log('🚀 EXECUTING ROUTE PREDICTION BENCHMARK SUITE');
  console.log('======================================================================\n');

  // ── 1. Benchmark on REAL Dataset ──
  console.log('>>> [1/2] BENCHMARKING ON REAL DATASET (data/detections.json)...');
  const realDataPath = path.join(__dirname, '..', 'data', 'detections.json');
  const realTopology = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'camera_network.json'), 'utf8'));
  const realSplit = prepareDataset(realDataPath);
  const realModel = trainModels(realSplit.trainJourneys, realTopology);

  console.log(`  Real dataset journeys: ${realSplit.allJourneys.length} (Train: ${realSplit.trainJourneys.length}, Test: ${realSplit.testJourneys.length})`);
  console.log(`  Real testable prediction events: ${realSplit.testEvents.length}`);

  const realResults = {
    baseline0: evaluatePredictor('Global Most Common', Predictors.baseline0, realSplit.testEvents, realModel),
    baseline1: evaluatePredictor('Most Common Outgoing', Predictors.baseline1, realSplit.testEvents, realModel),
    modelA: evaluatePredictor('First-Order Markov', Predictors.modelA_firstOrder, realSplit.testEvents, realModel),
    modelB: evaluatePredictor('Second-Order Markov', Predictors.modelB_secondOrder, realSplit.testEvents, realModel),
    modelC: evaluatePredictor('Time-Aware Model', Predictors.modelC_timeAware, realSplit.testEvents, realModel),
    modelD: evaluatePredictor('Direction-Aware Model', Predictors.modelD_directionAware, realSplit.testEvents, realModel),
    modelE: evaluatePredictor('Full Contextual Model', Predictors.modelE_fullContextual, realSplit.testEvents, realModel),
  };
  const realEta = evaluateETA(realSplit.testEvents, realModel, realTopology);

  // ── 2. Benchmark on SIMULATED Branching Network ──
  console.log('\n>>> [2/2] BENCHMARKING ON SIMULATION DATASET (data/simulation/detections.json)...');
  const simDataPath = path.join(__dirname, '..', 'data', 'simulation', 'detections.json');
  const simTopology = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'simulation', 'camera_network.json'), 'utf8'));
  const simSplit = prepareDataset(simDataPath);
  const simModel = trainModels(simSplit.trainJourneys, simTopology);

  console.log(`  Simulation dataset journeys: ${simSplit.allJourneys.length} (Train: ${simSplit.trainJourneys.length}, Test: ${simSplit.testJourneys.length})`);
  console.log(`  Simulation testable prediction events: ${simSplit.testEvents.length}`);

  const simResults = {
    baseline0: evaluatePredictor('Global Most Common', Predictors.baseline0, simSplit.testEvents, simModel),
    baseline1: evaluatePredictor('Most Common Outgoing', Predictors.baseline1, simSplit.testEvents, simModel),
    modelA: evaluatePredictor('First-Order Markov', Predictors.modelA_firstOrder, simSplit.testEvents, simModel),
    modelB: evaluatePredictor('Second-Order Markov', Predictors.modelB_secondOrder, simSplit.testEvents, simModel),
    modelC: evaluatePredictor('Time-Aware Model', Predictors.modelC_timeAware, simSplit.testEvents, simModel),
    modelD: evaluatePredictor('Direction-Aware Model', Predictors.modelD_directionAware, simSplit.testEvents, simModel),
    modelE: evaluatePredictor('Full Contextual Model', Predictors.modelE_fullContextual, simSplit.testEvents, simModel),
  };
  const simEta = evaluateETA(simSplit.testEvents, simModel, simTopology);

  return {
    real: { split: realSplit, results: realResults, eta: realEta },
    sim: { split: simSplit, results: simResults, eta: simEta },
  };
}

if (require.main === module) {
  const res = runAllBenchmarks();

  console.log('\n======================================================================');
  console.log('📊 BENCHMARK SUMMARY (SIMULATION BRANCHING NETWORK - 12 CAMERAS)');
  console.log('======================================================================');
  console.log('MODEL                     | TOP-1   | TOP-2   | TOP-3   | MRR   | COVERAGE');
  console.log('--------------------------|---------|---------|---------|-------|---------');
  Object.values(res.sim.results).forEach((r) => {
    console.log(
      `${r.name.padEnd(25)} | ${String(r.top1 + '%').padStart(7)} | ${String(r.top2 + '%').padStart(7)} | ${String(r.top3 + '%').padStart(7)} | ${String(r.mrr).padStart(5)} | ${String(r.coverage + '%').padStart(8)}`
    );
  });

  console.log('\n======================================================================');
  console.log('📊 ETA BENCHMARK RESULTS (SIMULATION)');
  console.log('======================================================================');
  console.log('METHOD                    | MAE     | MEDIAN  | P90     | COUNT');
  console.log('--------------------------|---------|---------|---------|------');
  console.log(`OSRM Network Duration     | ${String(res.sim.eta.osrm.mae + 's').padStart(7)} | ${String(res.sim.eta.osrm.median + 's').padStart(7)} | ${String(res.sim.eta.osrm.p90 + 's').padStart(7)} | ${res.sim.eta.osrm.count}`);
  console.log(`Historical Median Travel  | ${String(res.sim.eta.historicalMedian.mae + 's').padStart(7)} | ${String(res.sim.eta.historicalMedian.median + 's').padStart(7)} | ${String(res.sim.eta.historicalMedian.p90 + 's').padStart(7)} | ${res.sim.eta.historicalMedian.count}`);
  console.log(`Contextual Blended (60/40)| ${String(res.sim.eta.contextual.mae + 's').padStart(7)} | ${String(res.sim.eta.contextual.median + 's').padStart(7)} | ${String(res.sim.eta.contextual.p90 + 's').padStart(7)} | ${res.sim.eta.contextual.count}`);
}

module.exports = {
  prepareDataset,
  trainModels,
  Predictors,
  evaluatePredictor,
  evaluateETA,
  runAllBenchmarks,
};
