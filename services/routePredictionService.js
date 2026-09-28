'use strict';

/**
 * Road-Aware Vehicle Route Prediction Engine
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 *
 * Distinct from the Road Routing Engine.
 * 1. Prediction Engine: Predicts likely next camera/corridor using historical ANPR transitions,
 *    temporal patterns, road topology feasibility, directional alignment, and trip context.
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
 * Builds a Markov transition frequency matrix and corridor transitions from historical journeys.
 *
 * @param {Array<object>} journeys Array of valid journey sessions
 * @returns {object}
 */
function buildTransitionModel(journeys = []) {
  const singleTransitions = {}; // fromCam -> { toCam: count }
  const corridorTransitions = {}; // fromCam+midCam -> { toCam: count }
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

      // Travel time collection
      const t1 = new Date(sightings[i - 1].timestamp).getTime();
      const t2 = new Date(sightings[i].timestamp).getTime();
      if (!isNaN(t1) && !isNaN(t2) && t2 > t1) {
        const sec = Math.round((t2 - t1) / 1000);
        if (sec > 0 && sec < 7200) {
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

  return { singleTransitions, corridorTransitions, cameraTravelTimes };
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

  // 1. Build transition model from historical valid journeys
  const { singleTransitions, corridorTransitions, cameraTravelTimes } = buildTransitionModel(allJourneys);

  // 2. Identify candidate cameras (all network cameras except current)
  const allNetworkCameras = (topology.cameras && topology.cameras.length > 0)
    ? topology.cameras
    : [
        { standardId: 'CAM_01', name: 'Junction A', roadName: 'Chennai Central', allowedTravelDirection: 'northbound' },
        { standardId: 'CAM_02', name: 'Junction B', roadName: 'T. Nagar', allowedTravelDirection: 'eastbound' },
        { standardId: 'CAM_03', name: 'Junction C', roadName: 'Perambur', allowedTravelDirection: 'southbound' },
        { standardId: 'CAM_04', name: 'Highway Entry', roadName: 'Guindy', allowedTravelDirection: 'westbound' },
      ];

  const candidateCameras = allNetworkCameras.filter((c) => c.standardId !== currentCamId);

  // Direction of movement in current session
  let prevCamId = null;
  if (sightings.length >= 2) {
    prevCamId = formatCamId(sightings[sightings.length - 2].cameraId);
  }
  const detectedDirection = (lastSighting.direction || currentNode.allowedTravelDirection || '').toLowerCase();

  // 3. Score each candidate camera
  const scoredCandidates = [];

  const transitionsFromCurrent = singleTransitions[currentCamId] || {};
  let totalTransitions = 0;
  Object.values(transitionsFromCurrent).forEach((cnt) => { totalTransitions += cnt; });

  const corridorKey = prevCamId ? `${prevCamId}->${currentCamId}` : null;
  const corridorFromCurrent = corridorKey && corridorTransitions[corridorKey] ? corridorTransitions[corridorKey] : {};
  let totalCorridorTransitions = 0;
  Object.values(corridorFromCurrent).forEach((cnt) => { totalCorridorTransitions += cnt; });

  candidateCameras.forEach((candidate) => {
    const candId = candidate.standardId;
    let score = 0;
    const scoreFactors = [];

    // Factor A: Historical 1st-order transition probability
    const histCount = transitionsFromCurrent[candId] || 0;
    if (totalTransitions > 0 && histCount > 0) {
      const histProb = histCount / totalTransitions;
      score += histProb * 0.45;
      scoreFactors.push({ factor: 'HISTORICAL_TRANSITION', value: histProb, count: histCount });
    } else {
      // Baseline prior for connected nodes
      score += 0.08;
      scoreFactors.push({ factor: 'BASELINE_NETWORK_PRIOR', value: 0.08 });
    }

    // Factor B: 2nd-order corridor continuity (momentum / straight corridor)
    if (corridorKey && totalCorridorTransitions > 0) {
      const corrCount = corridorFromCurrent[candId] || 0;
      if (corrCount > 0) {
        const corrProb = corrCount / totalCorridorTransitions;
        score += corrProb * 0.25;
        scoreFactors.push({ factor: 'CORRIDOR_CONTINUITY', value: corrProb, count: corrCount });
      }
    }

    // Factor C: Road topology feasibility & distance penalty
    const topoRouteKey = `${currentCamId}:${candId}`;
    const topoRoute = topology.routes ? topology.routes[topoRouteKey] : null;

    if (topoRoute && topoRoute.roadAligned) {
      // Shorter physical road distances receive feasibility bonus
      const distBonus = Math.max(0.05, 0.20 - (topoRoute.distanceKm / 20.0) * 0.15);
      score += distBonus;
      scoreFactors.push({ factor: 'TOPOLOGICAL_ROAD_CONNECTIVITY', distanceKm: topoRoute.distanceKm, bonus: distBonus });
    } else {
      score += 0.02;
    }

    // Factor D: Directional feasibility
    // Penalize direct turnaround / U-turn unless no other exit exists
    if (prevCamId && candId === prevCamId) {
      score *= 0.35; // Significant penalty for immediate turnaround
      scoreFactors.push({ factor: 'REVERSAL_PENALTY', penalty: 0.35 });
    }

    // Directional alignment with camera's outgoing directions
    const outgoing = currentNode.possibleOutgoingRoadDirections || [];
    if (outgoing.length > 0 && candidate.allowedTravelDirection) {
      if (outgoing.includes(candidate.allowedTravelDirection.toLowerCase())) {
        score += 0.10;
        scoreFactors.push({ factor: 'DIRECTIONAL_ALIGNMENT', bonus: 0.10 });
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
      scoreFactors,
    });
  });

  // Sort by raw score descending
  scoredCandidates.sort((a, b) => b.rawScore - a.rawScore);

  // Normalize scores to valid probabilities summing to 1.0
  const sumRawScores = scoredCandidates.reduce((sum, c) => sum + c.rawScore, 0) || 1;
  scoredCandidates.forEach((cand) => {
    cand.probability = Number((cand.rawScore / sumRawScores).toFixed(2));
  });

  // Ensure probabilities sum exactly to 1.0
  const probSum = scoredCandidates.reduce((sum, c) => sum + c.probability, 0);
  if (scoredCandidates.length > 0 && Math.abs(probSum - 1.0) > 0.001) {
    scoredCandidates[0].probability = Number((scoredCandidates[0].probability + (1.0 - probSum)).toFixed(2));
  }

  // 4. Resolve Road Routing Geometry and ETA for Top Candidates
  const topLimit = options.limit || 3;
  const topCandidates = scoredCandidates.slice(0, topLimit);

  const resolvedCandidates = [];
  for (let idx = 0; idx < topCandidates.length; idx++) {
    const cand = topCandidates[idx];
    const isPrimary = idx === 0;

    // Obtain actual road geometry via Road Routing Engine
    const roadRoute = await getRoadRoute(currentCamId, cand.standardId);

    // Compute ETA: Combine historical travel time + road routing duration
    const histKey = `${currentCamId}->${cand.standardId}`;
    const historicalTimes = cameraTravelTimes[histKey] || [];
    let avgHistoricalSec = null;
    if (historicalTimes.length > 0) {
      const sumTimes = historicalTimes.reduce((s, t) => s + t, 0);
      avgHistoricalSec = Math.round(sumTimes / historicalTimes.length);
    }

    let etaSeconds = roadRoute.durationSeconds;
    if (avgHistoricalSec !== null) {
      // 60% historical observation + 40% road network calculation
      etaSeconds = Math.round(avgHistoricalSec * 0.6 + roadRoute.durationSeconds * 0.4);
    }

    const etaMins = Math.floor(etaSeconds / 60);
    const etaRemSec = etaSeconds % 60;
    const etaFormatted = etaMins > 0 ? `${etaMins}m ${etaRemSec}s` : `${etaSeconds}s`;

    // Confidence metric: based on sample evidence and prediction margin
    const confidence = isPrimary && cand.probability > 0.45 ? 0.88 : (cand.probability > 0.3 ? 0.72 : 0.55);

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
      confidence,
      isPrimary,
      etaSeconds,
      etaFormatted,
      historicalSampleCount: cand.historicalCount,
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

  // 5. Multi-Hop Route Prediction (Step 11)
  // If primary candidate exists, predict 2nd hop: current -> primary -> next
  let multiHopRoute = null;
  if (resolvedCandidates.length > 0 && options.includeMultiHop !== false) {
    const primary = resolvedCandidates[0];
    const primaryId = primary.standardId;

    // What is next after primary?
    const transitionsFromPrimary = singleTransitions[primaryId] || {};
    let bestSecondHop = null;
    let bestSecondCount = 0;
    Object.entries(transitionsFromPrimary).forEach(([nextId, count]) => {
      if (nextId !== currentCamId && count > bestSecondCount) {
        bestSecondCount = count;
        bestSecondHop = nextId;
      }
    });

    if (!bestSecondHop) {
      // Fallback to closest remaining node in topology
      const remaining = allNetworkCameras.filter((c) => c.standardId !== currentCamId && c.standardId !== primaryId);
      if (remaining.length > 0) {
        bestSecondHop = remaining[0].standardId;
      }
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

  // 1. Build transition model from historical valid journeys
  const { singleTransitions, corridorTransitions, cameraTravelTimes } = buildTransitionModel(allJourneys);

  // 2. Identify candidate cameras
  const allNetworkCameras = (topology.cameras && topology.cameras.length > 0)
    ? topology.cameras
    : [
        { standardId: 'CAM_01', name: 'Junction A', roadName: 'Chennai Central', allowedTravelDirection: 'northbound' },
        { standardId: 'CAM_02', name: 'Junction B', roadName: 'T. Nagar', allowedTravelDirection: 'eastbound' },
        { standardId: 'CAM_03', name: 'Junction C', roadName: 'Perambur', allowedTravelDirection: 'southbound' },
        { standardId: 'CAM_04', name: 'Highway Entry', roadName: 'Guindy', allowedTravelDirection: 'westbound' },
      ];

  const candidateCameras = allNetworkCameras.filter((c) => c.standardId !== currentCamId);

  let prevCamId = null;
  if (sightings.length >= 2) {
    prevCamId = formatCamId(sightings[sightings.length - 2].cameraId);
  }
  const detectedDirection = (lastSighting.direction || currentNode.allowedTravelDirection || '').toLowerCase();

  // 3. Score each candidate camera
  const scoredCandidates = [];

  const transitionsFromCurrent = singleTransitions[currentCamId] || {};
  let totalTransitions = 0;
  Object.values(transitionsFromCurrent).forEach((cnt) => { totalTransitions += cnt; });

  const corridorKey = prevCamId ? `${prevCamId}->${currentCamId}` : null;
  const corridorFromCurrent = corridorKey && corridorTransitions[corridorKey] ? corridorTransitions[corridorKey] : {};
  let totalCorridorTransitions = 0;
  Object.values(corridorFromCurrent).forEach((cnt) => { totalCorridorTransitions += cnt; });

  candidateCameras.forEach((candidate) => {
    const candId = candidate.standardId;
    let score = 0;
    const scoreFactors = [];

    // Factor A: Historical transition
    const histCount = transitionsFromCurrent[candId] || 0;
    if (totalTransitions > 0 && histCount > 0) {
      const histProb = histCount / totalTransitions;
      score += histProb * 0.45;
      scoreFactors.push({ factor: 'HISTORICAL_TRANSITION', value: histProb, count: histCount });
    } else {
      score += 0.08;
      scoreFactors.push({ factor: 'BASELINE_NETWORK_PRIOR', value: 0.08 });
    }

    // Factor B: 2nd-order corridor continuity
    if (corridorKey && totalCorridorTransitions > 0) {
      const corrCount = corridorFromCurrent[candId] || 0;
      if (corrCount > 0) {
        const corrProb = corrCount / totalCorridorTransitions;
        score += corrProb * 0.25;
        scoreFactors.push({ factor: 'CORRIDOR_CONTINUITY', value: corrProb, count: corrCount });
      }
    }

    // Factor C: Road topology feasibility & distance penalty
    const topoRouteKey = `${currentCamId}:${candId}`;
    const topoRoute = topology.routes ? topology.routes[topoRouteKey] : null;

    if (topoRoute && topoRoute.roadAligned) {
      const distBonus = Math.max(0.05, 0.20 - (topoRoute.distanceKm / 20.0) * 0.15);
      score += distBonus;
      scoreFactors.push({ factor: 'TOPOLOGICAL_ROAD_CONNECTIVITY', distanceKm: topoRoute.distanceKm, bonus: distBonus });
    } else {
      score += 0.02;
    }

    // Factor D: Directional feasibility
    if (prevCamId && candId === prevCamId) {
      score *= 0.35;
      scoreFactors.push({ factor: 'REVERSAL_PENALTY', penalty: 0.35 });
    }

    const outgoing = currentNode.possibleOutgoingRoadDirections || [];
    if (outgoing.length > 0 && candidate.allowedTravelDirection) {
      if (outgoing.includes(candidate.allowedTravelDirection.toLowerCase())) {
        score += 0.10;
        scoreFactors.push({ factor: 'DIRECTIONAL_ALIGNMENT', bonus: 0.10 });
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

  const topLimit = options.limit || 3;
  const topCandidates = scoredCandidates.slice(0, topLimit);

  const resolvedCandidates = [];
  for (let idx = 0; idx < topCandidates.length; idx++) {
    const cand = topCandidates[idx];
    const isPrimary = idx === 0;

    const roadRoute = getRoadRouteSync(currentCamId, cand.standardId);

    const histKey = `${currentCamId}->${cand.standardId}`;
    const historicalTimes = cameraTravelTimes[histKey] || [];
    let avgHistoricalSec = null;
    if (historicalTimes.length > 0) {
      const sumTimes = historicalTimes.reduce((s, t) => s + t, 0);
      avgHistoricalSec = Math.round(sumTimes / historicalTimes.length);
    }

    let etaSeconds = roadRoute.durationSeconds;
    if (avgHistoricalSec !== null) {
      etaSeconds = Math.round(avgHistoricalSec * 0.6 + roadRoute.durationSeconds * 0.4);
    }

    const etaMins = Math.floor(etaSeconds / 60);
    const etaRemSec = etaSeconds % 60;
    const etaFormatted = etaMins > 0 ? `${etaMins}m ${etaRemSec}s` : `${etaSeconds}s`;
    const confidence = isPrimary && cand.probability > 0.45 ? 0.88 : (cand.probability > 0.3 ? 0.72 : 0.55);

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
      confidence,
      isPrimary,
      etaSeconds,
      etaFormatted,
      historicalSampleCount: cand.historicalCount,
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

  let multiHopRoute = null;
  if (resolvedCandidates.length > 0 && options.includeMultiHop !== false) {
    const primary = resolvedCandidates[0];
    const primaryId = primary.standardId;

    const transitionsFromPrimary = singleTransitions[primaryId] || {};
    let bestSecondHop = null;
    let bestSecondCount = 0;
    Object.entries(transitionsFromPrimary).forEach(([nextId, count]) => {
      if (nextId !== currentCamId && count > bestSecondCount) {
        bestSecondCount = count;
        bestSecondHop = nextId;
      }
    });

    if (!bestSecondHop) {
      const remaining = allNetworkCameras.filter((c) => c.standardId !== currentCamId && c.standardId !== primaryId);
      if (remaining.length > 0) {
        bestSecondHop = remaining[0].standardId;
      }
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
  buildTransitionModel,
  predictNextCameras,
  predictNextCamerasSync,
};
