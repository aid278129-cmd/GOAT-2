'use strict';

/**
 * Road-Aware Vehicle Route Prediction Engine
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 *
 * Distinct from the Road Routing Engine:
 * 1. Prediction Engine: Predicts likely next camera/corridor using empirical Markov transitions,
 *    temporal time-of-day patterns, road topology feasibility, directional alignment, and trip context.
 * 2. Routing Engine: Obtains physically plausible road geometry between the predicted nodes.
 */

const fs = require('fs');
const path = require('path');
const {
  resolveCameraNode,
  getRoadRoute,
  getRoadRouteSync,
  getRoadRouteThroughCameras,
  getRoadRouteThroughCamerasSync,
  loadCameraTopology,
} = require('./roadRoutingService');

const CAMERAS_FILE = path.join(__dirname, '..', 'data', 'cameras.json');

/**
 * Standardizes plate string.
 */
function normalizePlate(plate) {
  if (!plate || typeof plate !== 'string') return '';
  return plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

/**
 * Standardizes camera ID string (e.g. 1 -> "CAM_01").
 */
function formatCamId(id) {
  if (!id) return '';
  const num = parseInt(String(id).replace(/[^0-9]/g, ''), 10);
  if (isNaN(num)) return String(id);
  return `CAM_${String(num).padStart(2, '0')}`;
}

/**
 * Categorizes an ISO timestamp into uniform time-of-day buckets.
 */
function getTimeBucket(isoTimestamp) {
  if (!isoTimestamp) return 'DAY';
  const d = new Date(isoTimestamp);
  const hour = d.getUTCHours();
  if (hour >= 6 && hour < 11) return 'MORNING_PEAK';
  if (hour >= 11 && hour < 16) return 'MIDDAY';
  if (hour >= 16 && hour < 21) return 'EVENING_PEAK';
  return 'NIGHT';
}

/**
 * Builds a Markov transition frequency matrix and corridor transitions from historical journeys.
 *
 * @param {Array<object>} journeys Array of valid journey sessions
 * @returns {object}
 */
function buildTransitionModel(journeys = []) {
  const singleTransitions = {}; // fromCam -> { toCam: count }
  const corridorTransitions = {}; // `${prevCam}->${currCam}` -> { toCam: count }
  const timeTransitions = {}; // `${fromCam}:${bucket}` -> { toCam: count }
  const cameraTravelTimes = {}; // `${fromCam}->${toCam}` -> [seconds]

  journeys.forEach((journey) => {
    const sightings = journey.sessionSightings || journey.sightings || [];
    if (sightings.length < 2) return;

    for (let i = 1; i < sightings.length; i++) {
      const prev = formatCamId(sightings[i - 1].cameraId);
      const curr = formatCamId(sightings[i].cameraId);

      if (!prev || !curr || prev === curr) continue;

      // 1st order transition
      if (!singleTransitions[prev]) singleTransitions[prev] = {};
      singleTransitions[prev][curr] = (singleTransitions[prev][curr] || 0) + 1;

      // Time bucket transition
      const bucket = getTimeBucket(sightings[i - 1].timestamp);
      const timeKey = `${prev}:${bucket}`;
      if (!timeTransitions[timeKey]) timeTransitions[timeKey] = {};
      timeTransitions[timeKey][curr] = (timeTransitions[timeKey][curr] || 0) + 1;

      // Travel time collection
      const t1 = new Date(sightings[i - 1].timestamp).getTime();
      const t2 = new Date(sightings[i].timestamp).getTime();
      if (!isNaN(t1) && !isNaN(t2) && t2 > t1) {
        const sec = Math.round((t2 - t1) / 1000);
        if (sec > 0 && sec < 10800) {
          const key = `${prev}->${curr}`;
          if (!cameraTravelTimes[key]) cameraTravelTimes[key] = [];
          cameraTravelTimes[key].push(sec);
        }
      }

      // 2nd order corridor transition
      if (i >= 2) {
        const prevPrev = formatCamId(sightings[i - 2].cameraId);
        if (prevPrev && prevPrev !== prev) {
          const corridorKey = `${prevPrev}->${prev}`;
          if (!corridorTransitions[corridorKey]) corridorTransitions[corridorKey] = {};
          corridorTransitions[corridorKey][curr] = (corridorTransitions[corridorKey][curr] || 0) + 1;
        }
      }
    }
  });

  const medianTravelTimes = {};
  Object.entries(cameraTravelTimes).forEach(([pair, arr]) => {
    arr.sort((a, b) => a - b);
    const mid = Math.floor(arr.length / 2);
    medianTravelTimes[pair] = arr.length % 2 !== 0 ? arr[mid] : Math.round((arr[mid - 1] + arr[mid]) / 2);
  });

  return { singleTransitions, corridorTransitions, timeTransitions, cameraTravelTimes, medianTravelTimes };
}

let _fleetJourneysCache = null;

/**
 * Returns fleet-wide historical journeys for transition model building.
 * If provided journeys contain fewer than 2 multi-camera sessions, falls back to loading detections.json.
 */
function getFleetJourneys(providedJourneys = []) {
  const multiCount = providedJourneys.filter((j) => (j.sessionSightings || j.sightings || []).length >= 2).length;
  if (multiCount >= 2) return providedJourneys;

  if (_fleetJourneysCache && _fleetJourneysCache.length > 0) {
    return _fleetJourneysCache;
  }

  const detPath = path.join(__dirname, '..', 'data', 'detections.json');
  if (fs.existsSync(detPath)) {
    try {
      const raw = JSON.parse(fs.readFileSync(detPath, 'utf8'));
      const { extractFleetJourneys } = require('./trajectoryService');
      const list = extractFleetJourneys(raw);
      _fleetJourneysCache = list;
      return list;
    } catch (e) {
      console.warn('⚠️ Could not load fleet detections for transition model:', e.message);
    }
  }
  return providedJourneys;
}

/**
 * Scores candidate next cameras using hierarchical empirical evidence and topology constraints.
 */
function scoreCandidateCameras(currentCamId, prevCamId, lastSighting, currentNode, topology, transitionModel) {
  const { singleTransitions, corridorTransitions, timeTransitions, medianTravelTimes } = transitionModel;

  const allNetworkCameras = (topology.cameras && topology.cameras.length > 0)
    ? topology.cameras
    : [
        { standardId: 'CAM_01', name: 'Junction A', roadName: 'Chennai Central', allowedTravelDirection: 'northbound' },
        { standardId: 'CAM_02', name: 'Junction B', roadName: 'T. Nagar', allowedTravelDirection: 'eastbound' },
        { standardId: 'CAM_03', name: 'Junction C', roadName: 'Perambur', allowedTravelDirection: 'southbound' },
        { standardId: 'CAM_04', name: 'Highway Entry', roadName: 'Guindy', allowedTravelDirection: 'westbound' },
      ];

  const candidateCameras = allNetworkCameras.filter((c) => c.standardId !== currentCamId);

  const transitionsFromCurrent = singleTransitions[currentCamId] || {};
  let totalTransitions = 0;
  Object.values(transitionsFromCurrent).forEach((cnt) => { totalTransitions += cnt; });

  const corridorKey = prevCamId ? `${prevCamId}->${currentCamId}` : null;
  const corridorFromCurrent = corridorKey && corridorTransitions[corridorKey] ? corridorTransitions[corridorKey] : {};
  let totalCorridorTransitions = 0;
  Object.values(corridorFromCurrent).forEach((cnt) => { totalCorridorTransitions += cnt; });

  const timeBucket = getTimeBucket(lastSighting.timestamp);
  const timeKey = `${currentCamId}:${timeBucket}`;
  const timeTransObj = timeTransitions[timeKey] || {};
  const totalTimeTrans = Object.values(timeTransObj).reduce((s, v) => s + v, 0);

  const scoredCandidates = [];

  candidateCameras.forEach((candidate) => {
    const candId = candidate.standardId;
    let score = 0;
    const scoreFactors = [];
    let method = 'FIRST_ORDER_MARKOV';

    // Factor A: Historical 1st-order transition probability
    const histCount = transitionsFromCurrent[candId] || 0;
    let rawFirstOrderProb = 0;

    if (totalTransitions > 0 && histCount > 0) {
      rawFirstOrderProb = histCount / totalTransitions;
      score += rawFirstOrderProb * 0.55;
      scoreFactors.push({ factor: 'HISTORICAL_1ST_ORDER', value: rawFirstOrderProb, count: histCount, total: totalTransitions });
    } else if (totalTransitions > 0) {
      score += 0.01; // Candidate was never observed historically from this node
      scoreFactors.push({ factor: 'UNOBSERVED_EDGE_PENALTY', value: 0.01 });
    } else {
      score += 0.10; // Zero historical outgoing data; use baseline prior
      scoreFactors.push({ factor: 'BASELINE_NETWORK_PRIOR', value: 0.10 });
      method = 'TOPOLOGY_PRIOR';
    }

    // Factor B: 2nd-order corridor continuity
    let corrCount = 0;
    if (corridorKey && totalCorridorTransitions > 0) {
      corrCount = corridorFromCurrent[candId] || 0;
      if (corrCount > 0) {
        const corrProb = corrCount / totalCorridorTransitions;
        score += corrProb * 0.30;
        scoreFactors.push({ factor: 'CORRIDOR_CONTINUITY_2ND_ORDER', value: corrProb, count: corrCount, total: totalCorridorTransitions });
        method = 'SECOND_ORDER_CONTEXTUAL';
      }
    }

    // Factor C: Time-of-day alignment
    const timeCount = timeTransObj[candId] || 0;
    if (totalTimeTrans > 0 && timeCount > 0) {
      const timeProb = timeCount / totalTimeTrans;
      score += timeProb * 0.15;
      scoreFactors.push({ factor: 'TIME_OF_DAY_ALIGNMENT', value: timeProb, count: timeCount, bucket: timeBucket });
    }

    // Factor D: Road topology feasibility & distance penalty
    const topoRouteKey = `${currentCamId}:${candId}`;
    const topoRoute = topology.routes ? topology.routes[topoRouteKey] : null;

    if (topoRoute && topoRoute.roadAligned) {
      const distBonus = Math.max(0.04, 0.15 - (topoRoute.distanceKm / 20.0) * 0.10);
      score += distBonus;
      scoreFactors.push({ factor: 'TOPOLOGICAL_ROAD_CONNECTIVITY', distanceKm: topoRoute.distanceKm, bonus: distBonus });
    } else {
      score += 0.02;
    }

    // Factor E: Directional feasibility & Reversal penalty
    let isReversal = false;
    if (prevCamId && candId === prevCamId) {
      isReversal = true;
      score *= 0.35; // Significant penalty for immediate turnaround
      scoreFactors.push({ factor: 'REVERSAL_PENALTY', penalty: 0.35 });
    }

    const outgoing = currentNode.possibleOutgoingRoadDirections || [];
    let isDirectionAligned = false;
    if (outgoing.length > 0 && candidate.allowedTravelDirection) {
      if (outgoing.includes(candidate.allowedTravelDirection.toLowerCase())) {
        isDirectionAligned = true;
        score += 0.08;
        scoreFactors.push({ factor: 'DIRECTIONAL_ALIGNMENT', bonus: 0.08 });
      }
    }

    scoredCandidates.push({
      cameraId: candId,
      standardId: candId,
      cameraName: candidate.name,
      location: candidate.roadName || candidate.location || '',
      latitude: candidate.latitude,
      longitude: candidate.longitude,
      lat: candidate.latitude,
      lng: candidate.longitude,
      rawScore: score,
      historicalCount: histCount,
      corrCount,
      timeCount,
      timeBucket,
      topoDistanceKm: topoRoute ? topoRoute.distanceKm : null,
      topoVerified: topoRoute ? topoRoute.roadAligned : false,
      isReversal,
      isDirectionAligned,
      method,
      scoreFactors,
    });
  });

  scoredCandidates.sort((a, b) => b.rawScore - a.rawScore);

  const sumRawScores = scoredCandidates.reduce((sum, c) => sum + c.rawScore, 0) || 1;
  scoredCandidates.forEach((cand) => {
    cand.probability = Number((cand.rawScore / sumRawScores).toFixed(2));
  });

  const probSum = scoredCandidates.reduce((sum, c) => sum + c.probability, 0);
  if (scoredCandidates.length > 0 && Math.abs(probSum - 1.0) > 0.001) {
    scoredCandidates[0].probability = Number((scoredCandidates[0].probability + (1.0 - probSum)).toFixed(2));
  }

  return { scoredCandidates, totalTransitions, totalCorridorTransitions, medianTravelTimes };
}

/**
 * Predicts likely next camera nodes for a vehicle given its active journey session.
 *
 * @param {object} activeJourney Active journey session object
 * @param {Array<object>} allJourneys All valid historical journeys (for transition model)
 * @param {object} [options]
 * @returns {Promise<object>}
 */
async function predictNextCameras(activeJourney, allJourneys = [], options = {}) {
  const sightings = activeJourney ? (activeJourney.sessionSightings || activeJourney.sightings || []) : [];
  if (sightings.length === 0) {
    return {
      success: false,
      message: 'No sightings in active journey',
      currentCamera: null,
      nextCameras: [],
    };
  }

  const lastSighting = sightings[sightings.length - 1];
  const currentCamId = formatCamId(lastSighting.cameraId);
  const currentNode = resolveCameraNode(currentCamId);
  const topology = loadCameraTopology();

  if (!currentNode) {
    return {
      success: false,
      message: `Unknown current camera node: ${currentCamId}`,
      currentCamera: currentCamId,
      nextCameras: [],
    };
  }

  const fleetJourneys = getFleetJourneys(allJourneys);
  const transitionModel = buildTransitionModel(fleetJourneys);

  let prevCamId = null;
  if (sightings.length >= 2) {
    prevCamId = formatCamId(sightings[sightings.length - 2].cameraId);
  }
  const detectedDirection = (lastSighting.direction || currentNode.allowedTravelDirection || '').toLowerCase();

  const { scoredCandidates, totalTransitions, totalCorridorTransitions, medianTravelTimes } = scoreCandidateCameras(
    currentCamId,
    prevCamId,
    lastSighting,
    currentNode,
    topology,
    transitionModel
  );

  const topLimit = options.limit || 3;
  const topCandidates = scoredCandidates.slice(0, topLimit);

  const resolvedCandidates = [];
  for (let idx = 0; idx < topCandidates.length; idx++) {
    const cand = topCandidates[idx];
    const isPrimary = idx === 0;

    const roadRoute = await getRoadRoute(currentCamId, cand.standardId);

    const histKey = `${currentCamId}->${cand.standardId}`;
    const historicalMedianSec = medianTravelTimes[histKey] || null;

    let etaSeconds = roadRoute.durationSeconds;
    if (historicalMedianSec !== null) {
      etaSeconds = Math.round(historicalMedianSec * 0.6 + roadRoute.durationSeconds * 0.4);
    }

    const etaMins = Math.floor(etaSeconds / 60);
    const etaRemSec = etaSeconds % 60;
    const etaFormatted = etaMins > 0 ? `${etaMins}m ${etaRemSec}s` : `${etaSeconds}s`;

    let confidenceTier = 'LOW';
    let confidenceScore = 0.55;
    if (cand.historicalCount >= 5 && cand.probability >= 0.45) {
      confidenceTier = 'HIGH';
      confidenceScore = 0.90;
    } else if (cand.historicalCount >= 2 && cand.probability >= 0.25) {
      confidenceTier = 'MEDIUM';
      confidenceScore = 0.75;
    } else if (cand.historicalCount === 0) {
      confidenceTier = 'INSUFFICIENT_DATA';
      confidenceScore = 0.40;
    }

    resolvedCandidates.push({
      cameraId: cand.standardId,
      standardId: cand.standardId,
      cameraName: cand.cameraName,
      location: cand.location,
      latitude: cand.latitude,
      longitude: cand.longitude,
      lat: cand.latitude,
      lng: cand.longitude,
      probability: cand.probability,
      confidence: confidenceTier,
      confidenceTier,
      confidenceScore,
      isPrimary,
      etaSeconds,
      etaFormatted,
      historicalSampleCount: cand.historicalCount,
      evidence: {
        firstOrderTransitions: cand.historicalCount,
        totalOutgoingTransitions: totalTransitions,
        rawFirstOrderProbability: totalTransitions > 0 ? Number((cand.historicalCount / totalTransitions).toFixed(4)) : null,
        corridorTransitions: cand.corrCount || 0,
        totalCorridorTransitions,
        timeBucket: cand.timeBucket,
        timeBucketTransitions: cand.timeCount || 0,
        topologyDistanceKm: cand.topoDistanceKm,
        topologyVerified: cand.topoVerified,
        directionCompatible: !cand.isReversal && cand.isDirectionAligned,
        historicalMedianTravelSeconds: historicalMedianSec,
        method: cand.method,
        scoreFactors: cand.scoreFactors,
      },
      route: {
        roadAligned: roadRoute.roadAligned,
        routingStatus: roadRoute.routingStatus,
        geometry: roadRoute.geometry,
        latLngs: roadRoute.latLngs,
        distanceMeters: roadRoute.distanceMeters,
        distanceKm: roadRoute.distanceKm,
        durationSeconds: roadRoute.durationSeconds,
        provider: roadRoute.provider,
      },
    });
  }

  // Multi-Hop Route Prediction
  let multiHopRoute = null;
  if (resolvedCandidates.length > 0 && options.includeMultiHop !== false) {
    const primary = resolvedCandidates[0];
    const primaryId = primary.standardId;

    const transitionsFromPrimary = transitionModel.singleTransitions[primaryId] || {};
    let bestSecondHop = null;
    let bestSecondCount = 0;
    Object.entries(transitionsFromPrimary).forEach(([nextId, count]) => {
      if (nextId !== currentCamId && count > bestSecondCount) {
        bestSecondCount = count;
        bestSecondHop = nextId;
      }
    });

    if (!bestSecondHop) {
      const remaining = topology.cameras.filter((c) => c.standardId !== currentCamId && c.standardId !== primaryId);
      if (remaining.length > 0) bestSecondHop = remaining[0].standardId;
    }

    if (bestSecondHop) {
      const multiHopSequence = [currentCamId, primaryId, bestSecondHop];
      const multiHopRoad = await getRoadRouteThroughCameras(multiHopSequence);
      multiHopRoute = {
        sequence: multiHopSequence,
        cameras: multiHopSequence.map((id) => resolveCameraNode(id)),
        totalDistanceKm: multiHopRoad.totalDistanceKm,
        totalDurationSeconds: multiHopRoad.totalDurationSeconds,
        roadAligned: multiHopRoad.roadAligned,
        geometry: multiHopRoad.geometry,
        latLngs: multiHopRoad.latLngs,
        legs: multiHopRoad.legs,
      };
    }
  }

  return {
    success: true,
    plate: normalizePlate(activeJourney.plate || (lastSighting && lastSighting.plate) || ''),
    activeJourneyId: activeJourney.journeyId,
    currentCamera: currentCamId,
    currentCameraName: currentNode.name,
    currentLocation: currentNode.roadName,
    currentLatitude: currentNode.latitude,
    currentLongitude: currentNode.longitude,
    detectedDirection,
    lastSeen: lastSighting.timestamp,
    nextCameras: resolvedCandidates,
    multiHopRoute,
  };
}

/**
 * Synchronous route prediction using pre-cached camera topology and routes.
 *
 * @param {object} activeJourney
 * @param {Array<object>} allJourneys
 * @param {object} [options]
 * @returns {object}
 */
function predictNextCamerasSync(activeJourney, allJourneys = [], options = {}) {
  const sightings = activeJourney ? (activeJourney.sessionSightings || activeJourney.sightings || []) : [];
  if (sightings.length === 0) {
    return {
      success: false,
      message: 'No sightings in active journey',
      currentCamera: null,
      nextCameras: [],
    };
  }

  const lastSighting = sightings[sightings.length - 1];
  const currentCamId = formatCamId(lastSighting.cameraId);
  const currentNode = resolveCameraNode(currentCamId);
  const topology = loadCameraTopology();

  if (!currentNode) {
    return {
      success: false,
      message: `Unknown current camera node: ${currentCamId}`,
      currentCamera: currentCamId,
      nextCameras: [],
    };
  }

  const fleetJourneys = getFleetJourneys(allJourneys);
  const transitionModel = buildTransitionModel(fleetJourneys);

  let prevCamId = null;
  if (sightings.length >= 2) {
    prevCamId = formatCamId(sightings[sightings.length - 2].cameraId);
  }
  const detectedDirection = (lastSighting.direction || currentNode.allowedTravelDirection || '').toLowerCase();

  const { scoredCandidates, totalTransitions, totalCorridorTransitions, medianTravelTimes } = scoreCandidateCameras(
    currentCamId,
    prevCamId,
    lastSighting,
    currentNode,
    topology,
    transitionModel
  );

  const topLimit = options.limit || 3;
  const topCandidates = scoredCandidates.slice(0, topLimit);

  const resolvedCandidates = [];
  for (let idx = 0; idx < topCandidates.length; idx++) {
    const cand = topCandidates[idx];
    const isPrimary = idx === 0;

    const roadRoute = getRoadRouteSync(currentCamId, cand.standardId);

    const histKey = `${currentCamId}->${cand.standardId}`;
    const historicalMedianSec = medianTravelTimes[histKey] || null;

    let etaSeconds = roadRoute.durationSeconds;
    if (historicalMedianSec !== null) {
      etaSeconds = Math.round(historicalMedianSec * 0.6 + roadRoute.durationSeconds * 0.4);
    }

    const etaMins = Math.floor(etaSeconds / 60);
    const etaRemSec = etaSeconds % 60;
    const etaFormatted = etaMins > 0 ? `${etaMins}m ${etaRemSec}s` : `${etaSeconds}s`;

    let confidenceTier = 'LOW';
    let confidenceScore = 0.55;
    if (cand.historicalCount >= 5 && cand.probability >= 0.45) {
      confidenceTier = 'HIGH';
      confidenceScore = 0.90;
    } else if (cand.historicalCount >= 2 && cand.probability >= 0.25) {
      confidenceTier = 'MEDIUM';
      confidenceScore = 0.75;
    } else if (cand.historicalCount === 0) {
      confidenceTier = 'INSUFFICIENT_DATA';
      confidenceScore = 0.40;
    }

    resolvedCandidates.push({
      cameraId: cand.standardId,
      standardId: cand.standardId,
      cameraName: cand.cameraName,
      location: cand.location,
      latitude: cand.latitude,
      longitude: cand.longitude,
      lat: cand.latitude,
      lng: cand.longitude,
      probability: cand.probability,
      confidence: confidenceTier,
      confidenceTier,
      confidenceScore,
      isPrimary,
      etaSeconds,
      etaFormatted,
      historicalSampleCount: cand.historicalCount,
      evidence: {
        firstOrderTransitions: cand.historicalCount,
        totalOutgoingTransitions: totalTransitions,
        rawFirstOrderProbability: totalTransitions > 0 ? Number((cand.historicalCount / totalTransitions).toFixed(4)) : null,
        corridorTransitions: cand.corrCount || 0,
        totalCorridorTransitions,
        timeBucket: cand.timeBucket,
        timeBucketTransitions: cand.timeCount || 0,
        topologyDistanceKm: cand.topoDistanceKm,
        topologyVerified: cand.topoVerified,
        directionCompatible: !cand.isReversal && cand.isDirectionAligned,
        historicalMedianTravelSeconds: historicalMedianSec,
        method: cand.method,
        scoreFactors: cand.scoreFactors,
      },
      route: {
        roadAligned: roadRoute.roadAligned,
        routingStatus: roadRoute.routingStatus,
        geometry: roadRoute.geometry,
        latLngs: roadRoute.latLngs,
        distanceMeters: roadRoute.distanceMeters,
        distanceKm: roadRoute.distanceKm,
        durationSeconds: roadRoute.durationSeconds,
        provider: roadRoute.provider,
      },
    });
  }

  // Multi-Hop Route Prediction
  let multiHopRoute = null;
  if (resolvedCandidates.length > 0 && options.includeMultiHop !== false) {
    const primary = resolvedCandidates[0];
    const primaryId = primary.standardId;

    const transitionsFromPrimary = transitionModel.singleTransitions[primaryId] || {};
    let bestSecondHop = null;
    let bestSecondCount = 0;
    Object.entries(transitionsFromPrimary).forEach(([nextId, count]) => {
      if (nextId !== currentCamId && count > bestSecondCount) {
        bestSecondCount = count;
        bestSecondHop = nextId;
      }
    });

    if (!bestSecondHop) {
      const remaining = topology.cameras.filter((c) => c.standardId !== currentCamId && c.standardId !== primaryId);
      if (remaining.length > 0) bestSecondHop = remaining[0].standardId;
    }

    if (bestSecondHop) {
      const multiHopSequence = [currentCamId, primaryId, bestSecondHop];
      const multiHopRoad = getRoadRouteThroughCamerasSync(multiHopSequence);
      multiHopRoute = {
        sequence: multiHopSequence,
        cameras: multiHopSequence.map((id) => resolveCameraNode(id)),
        totalDistanceKm: multiHopRoad.totalDistanceKm,
        totalDurationSeconds: multiHopRoad.totalDurationSeconds,
        roadAligned: multiHopRoad.roadAligned,
        geometry: multiHopRoad.geometry,
        latLngs: multiHopRoad.latLngs,
        legs: multiHopRoad.legs,
      };
    }
  }

  return {
    success: true,
    plate: normalizePlate(activeJourney.plate || (lastSighting && lastSighting.plate) || ''),
    activeJourneyId: activeJourney.journeyId,
    currentCamera: currentCamId,
    currentCameraName: currentNode.name,
    currentLocation: currentNode.roadName,
    currentLatitude: currentNode.latitude,
    currentLongitude: currentNode.longitude,
    detectedDirection,
    lastSeen: lastSighting.timestamp,
    nextCameras: resolvedCandidates,
    multiHopRoute,
  };
}

module.exports = {
  normalizePlate,
  formatCamId,
  getTimeBucket,
  buildTransitionModel,
  predictNextCameras,
  predictNextCamerasSync,
};
