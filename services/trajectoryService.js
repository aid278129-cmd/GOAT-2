'use strict';

/**
 * Trajectory Reconstruction and Analysis Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const fs = require('fs');
const path = require('path');
const { haversineDistance, calculateEstimatedSpeed } = require('./geoService');
const {
  getRoadRouteSync,
  getRoadRouteThroughCamerasSync,
  getRoadRouteThroughCameras,
  loadCameraTopology,
} = require('./roadRoutingService');
const {
  predictNextCamerasSync,
  predictNextCameras,
} = require('./routePredictionService');

const CONFIG_FILE = path.join(__dirname, '..', 'data', 'analytics_config.json');

/**
 * Loads analytics configuration with safe fallbacks.
 */
function loadAnalyticsConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const raw = fs.readFileSync(CONFIG_FILE, 'utf8');
      return JSON.parse(raw);
    }
  } catch (err) {
    console.warn('⚠️  Could not load analytics_config.json, using default config:', err.message);
  }
  return {
    maxReasonableSpeedKmh: 180,
    minSegmentTravelTimeSeconds: 1,
    routeLoopThreshold: 3,
    routeLoopWindowMinutes: 60,
    trajectorySessionGapMinutes: 30,
    minPlausibleJourneySpeedKmh: 2.0,
  };
}

/**
 * Standardizes license plate string for indexing and query matching.
 *
 * @param {string} plate
 * @returns {string}
 */
function normalizePlate(plate) {
  if (!plate || typeof plate !== 'string') return '';
  return plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
}

/**
 * Safely extracts latitude and longitude from a detection record,
 * supporting both (latitude, longitude) and legacy (lat, lng).
 *
 * @param {object} det
 * @returns {{lat: number, lng: number}|null}
 */
function getCoordinates(det) {
  if (!det || typeof det !== 'object') return null;

  const latRaw = det.latitude !== undefined && det.latitude !== null ? det.latitude : det.lat;
  const lngRaw = det.longitude !== undefined && det.longitude !== null ? det.longitude : det.lng;

  if (latRaw === undefined || latRaw === null || lngRaw === undefined || lngRaw === null) {
    return null;
  }

  const lat = Number(latRaw);
  const lng = Number(lngRaw);

  if (isNaN(lat) || isNaN(lng) || !isFinite(lat) || !isFinite(lng)) {
    return null;
  }

  // Basic boundary check
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    return null;
  }

  // Treat (0, 0) as invalid/missing GPS coordinate
  if (lat === 0 && lng === 0) {
    return null;
  }

  return { lat, lng };
}

/**
 * Retrieves all sightings matching the given plate, sorted chronologically ascending.
 * Does not mutate the source detection array.
 *
 * @param {string} plate
 * @param {Array<object>} detections
 * @returns {Array<object>}
 */
function getPlateSightings(plate, detections) {
  const norm = normalizePlate(plate);
  if (!norm || !Array.isArray(detections)) return [];

  const matched = detections.filter((d) => normalizePlate(d.plate) === norm);

  // Return a sorted shallow clone
  return [...matched].sort((a, b) => {
    const timeA = new Date(a.timestamp).getTime();
    const timeB = new Date(b.timestamp).getTime();
    const safeA = isNaN(timeA) ? 0 : timeA;
    const safeB = isNaN(timeB) ? 0 : timeB;
    return safeA - safeB;
  });
}

/**
 * Calculates segments between consecutive sightings.
 *
 * @param {Array<object>} sightings
 * @param {object} [config]
 * @returns {Array<object>}
 */
function calculateTrajectorySegments(sightings, config = null) {
  if (!Array.isArray(sightings) || sightings.length < 2) {
    return [];
  }

  const cfg = config || loadAnalyticsConfig();
  const maxSpeed = Number(cfg.maxReasonableSpeedKmh) || 180;
  const segments = [];

  for (let i = 1; i < sightings.length; i++) {
    const prev = sightings[i - 1];
    const curr = sightings[i];

    const validationIssues = [];

    // Parse timestamps
    const prevTime = new Date(prev.timestamp).getTime();
    const currTime = new Date(curr.timestamp).getTime();

    let travelTimeSeconds = null;
    let travelMins = 0;

    if (isNaN(prevTime) || isNaN(currTime)) {
      validationIssues.push('MALFORMED_TIMESTAMP');
    } else {
      const diffMs = currTime - prevTime;
      if (diffMs < 0) {
        validationIssues.push('NEGATIVE_TRAVEL_TIME');
      } else if (diffMs === 0) {
        validationIssues.push('ZERO_TRAVEL_TIME');
        travelTimeSeconds = 0;
        travelMins = 0;
      } else {
        travelTimeSeconds = Math.round(diffMs / 1000);
        travelMins = Math.round(diffMs / 60000);
      }
    }

    // Camera identification
    const prevCamId = prev.cameraId !== undefined ? String(prev.cameraId) : '';
    const currCamId = curr.cameraId !== undefined ? String(curr.cameraId) : '';
    const isSameCamera = prevCamId && currCamId && prevCamId === currCamId;

    if (isSameCamera) {
      validationIssues.push('SAME_CAMERA_STATIONARY');
    }

    // Coordinates and distance
    const prevCoords = getCoordinates(prev);
    const currCoords = getCoordinates(curr);

    let distanceKm = null;
    if (!prevCoords || !currCoords) {
      validationIssues.push('MISSING_COORDINATES');
    } else {
      distanceKm = haversineDistance(
        prevCoords.lat,
        prevCoords.lng,
        currCoords.lat,
        currCoords.lng
      );
    }

    // Speed calculation
    let estimatedSpeedKmh = null;
    if (
      distanceKm !== null &&
      travelTimeSeconds !== null &&
      travelTimeSeconds > 0 &&
      !isSameCamera
    ) {
      estimatedSpeedKmh = calculateEstimatedSpeed(distanceKm, travelTimeSeconds);
    }

    // Anomaly checks
    let anomaly = null;
    let validForAnalytics = false;

    if (validationIssues.length === 0 && estimatedSpeedKmh !== null) {
      if (estimatedSpeedKmh > maxSpeed) {
        validForAnalytics = false;
        anomaly = {
          type: 'IMPOSSIBLE_TRAVEL',
          reason: `Calculated travel speed (${estimatedSpeedKmh.toFixed(
            1
          )} km/h) exceeds configured maximum (${maxSpeed} km/h)`,
          configuredMaxKmh: maxSpeed,
          calculatedSpeedKmh: Number(estimatedSpeedKmh.toFixed(2)),
        };
      } else {
        validForAnalytics = true;
      }
    }

    const prevName = prev.cameraName || prev.cameraLocation || `CAM-${prev.cameraId}`;
    const currName = curr.cameraName || curr.cameraLocation || `CAM-${curr.cameraId}`;

    segments.push({
      index: i,
      from: prevName, // legacy compatibility
      to: currName, // legacy compatibility
      fromCameraId: prev.cameraId,
      fromCameraName: prevName,
      fromLocation: prev.location || prev.cameraLocation || '',
      toCameraId: curr.cameraId,
      toCameraName: currName,
      toLocation: curr.location || curr.cameraLocation || '',
      fromTimestamp: prev.timestamp,
      toTimestamp: curr.timestamp,
      travelTimeSeconds,
      travelMins, // legacy compatibility
      distanceKm: distanceKm !== null ? Number(distanceKm.toFixed(2)) : null,
      rawDistanceKm: distanceKm,
      estimatedSpeedKmh: estimatedSpeedKmh !== null ? Number(estimatedSpeedKmh.toFixed(2)) : null,
      validForAnalytics,
      isSameCamera,
      validationIssues,
      anomaly,
    });
  }

  return segments;
}

/**
 * Calculates complete summary metrics for a vehicle's trajectory.
 *
 * @param {Array<object>} sightings
 * @param {Array<object>} segments
 * @returns {object}
 */
function calculateTrajectorySummary(sightings, segments = []) {
  const totalSightings = Array.isArray(sightings) ? sightings.length : 0;
  if (totalSightings === 0) {
    return {
      totalSightings: 0,
      totalDistanceKm: 0,
      totalTravelTimeSeconds: 0,
      totalTravelTimeMinutes: 0,
      averageJourneySpeedKmh: null,
      validSegmentCount: 0,
      invalidSegmentCount: 0,
      firstSeen: null,
      lastSeen: null,
    };
  }

  const firstSeen = sightings[0].timestamp || null;
  const lastSeen = sightings[sightings.length - 1].timestamp || null;

  let totalTravelTimeSeconds = 0;
  if (firstSeen && lastSeen) {
    const tStart = new Date(firstSeen).getTime();
    const tEnd = new Date(lastSeen).getTime();
    if (!isNaN(tStart) && !isNaN(tEnd) && tEnd >= tStart) {
      totalTravelTimeSeconds = Math.round((tEnd - tStart) / 1000);
    }
  }

  const totalTravelTimeMinutes = Number((totalTravelTimeSeconds / 60).toFixed(1));

  let totalDistanceKm = 0;
  let validDistanceKm = 0;
  let validTravelTimeSec = 0;
  let validSegmentCount = 0;
  let invalidSegmentCount = 0;

  segments.forEach((seg) => {
    if (seg.distanceKm !== null && !seg.isSameCamera) {
      totalDistanceKm += seg.distanceKm;
    }

    if (seg.validForAnalytics) {
      validSegmentCount++;
      validDistanceKm += seg.distanceKm || 0;
      validTravelTimeSec += seg.travelTimeSeconds || 0;
    } else {
      invalidSegmentCount++;
    }
  });

  // Calculate average journey speed based on valid segments (total valid distance / total valid travel time)
  let averageJourneySpeedKmh = null;
  if (validDistanceKm > 0 && validTravelTimeSec > 0) {
    const hours = validTravelTimeSec / 3600;
    averageJourneySpeedKmh = Number((validDistanceKm / hours).toFixed(2));
  }

  return {
    totalSightings,
    totalDistanceKm: Number(totalDistanceKm.toFixed(2)),
    totalTravelTimeSeconds,
    totalTravelTimeMinutes,
    averageJourneySpeedKmh,
    validSegmentCount,
    invalidSegmentCount,
    firstSeen,
    lastSeen,
  };
}

/**
 * Segments chronological vehicle sightings into discrete journeys/sessions.
 * Prevents joining observations from separate trips (e.g. 14-hour or 38-hour gaps)
 * into a single continuous trajectory.
 *
 * A vehicle trajectory is split when:
 * 1. Time gap between detections exceeds configurable threshold (trajectorySessionGapMinutes, default 30 min)
 * 2. Calculated movement speed between distant cameras is implausibly low (< minPlausibleJourneySpeedKmh, default 2 km/h across > 30m),
 *    indicating parked vehicle or distinct trips.
 *
 * @param {Array<object>} sightings Chronologically sorted sightings for a plate
 * @param {object} [config] Analytics configuration
 * @returns {Array<object>} Array of journey session objects
 */
function segmentSightingsIntoJourneys(sightings, config = null) {
  if (!Array.isArray(sightings) || sightings.length === 0) {
    return [];
  }

  const cfg = config || loadAnalyticsConfig();
  const maxGapMinutes = Number(cfg.trajectorySessionGapMinutes) || 30;
  const maxGapMs = maxGapMinutes * 60 * 1000;
  const minSpeedKmh = Number(cfg.minPlausibleJourneySpeedKmh) || 2.0;

  const rawGroups = [];
  let currentGroup = [sightings[0]];

  for (let i = 1; i < sightings.length; i++) {
    const prev = sightings[i - 1];
    const curr = sightings[i];

    const prevTime = new Date(prev.timestamp).getTime();
    const currTime = new Date(curr.timestamp).getTime();
    const diffMs = !isNaN(prevTime) && !isNaN(currTime) ? (currTime - prevTime) : 0;

    let split = false;
    let splitReason = '';

    // Condition 1: Time gap exceeds maximum configurable session gap
    if (diffMs > maxGapMs) {
      split = true;
      splitReason = `Time gap of ${(diffMs / 60000).toFixed(1)}m exceeds session threshold (${maxGapMinutes}m)`;
    } else if (diffMs > 1800000) { // > 30 minutes
      split = true;
      splitReason = `Large interval (${(diffMs / 60000).toFixed(1)}m) indicates separate trip`;
    }

    // Phase 5: Low speed is a SUPPORTING SIGNAL (congestion, signal delay), NOT a hard split
    if (!split && diffMs > 0 && prev.cameraId !== curr.cameraId) {
      const pCoord = getCoordinates(prev);
      const cCoord = getCoordinates(curr);
      if (pCoord && cCoord) {
        const segDist = haversineDistance(pCoord.lat, pCoord.lng, cCoord.lat, cCoord.lng) || 0;
        const segSpeed = segDist / (diffMs / 3600000);
        if (segSpeed < minSpeedKmh) {
          if (!curr.supportingSignals) curr.supportingSignals = [];
          curr.supportingSignals.push({
            type: 'LOW_SPEED_CONGESTION',
            speedKmh: Number(segSpeed.toFixed(2)),
            message: `Low transit speed (${segSpeed.toFixed(1)} km/h) treated as supporting signal for congestion, not journey split`,
          });
        }
      }
    }

    if (split) {
      rawGroups.push({ sightings: currentGroup, splitReason });
      currentGroup = [curr];
    } else {
      currentGroup.push(curr);
    }
  }

  if (currentGroup.length > 0) {
    rawGroups.push({ sightings: currentGroup, splitReason: null });
  }

  const plateNorm = normalizePlate(sightings[0].plate);

  // Transform each group into a full journey session
  return rawGroups.map((group, idx) => {
    const sGroup = group.sightings;
    const journeyId = `${plateNorm}-session-${idx + 1}`;
    const startTime = sGroup[0].timestamp;
    const endTime = sGroup[sGroup.length - 1].timestamp;

    const tStart = new Date(startTime).getTime();
    const tEnd = new Date(endTime).getTime();
    const durationSeconds = !isNaN(tStart) && !isNaN(tEnd) && tEnd >= tStart && sGroup.length >= 2
      ? Math.round((tEnd - tStart) / 1000)
      : 0;
    const durationMinutes = Number((durationSeconds / 60).toFixed(1));

    const segs = calculateTrajectorySegments(sGroup, cfg);
    const summary = calculateTrajectorySummary(sGroup, segs);

    // Resolve road routing geometry for this journey's camera sequence
    const camSequence = sGroup.map((s) => s.cameraId);
    let roadRoute = null;
    let roadDistanceKm = 0;
    if (sGroup.length >= 2) {
      roadRoute = getRoadRouteThroughCamerasSync(camSequence);
      roadDistanceKm = roadRoute && roadRoute.roadAligned ? roadRoute.totalDistanceKm : summary.totalDistanceKm;
    }

    // Phase 6: Single sighting active journey reports 0 distance and null speed
    const geodesicDistanceKm = sGroup.length >= 2 ? summary.totalDistanceKm : 0;
    const averageJourneySpeedKmh = sGroup.length >= 2 ? summary.averageJourneySpeedKmh : null;

    // Build points for this session
    const sessionPoints = sGroup.map((det, index) => {
      const coords = getCoordinates(det);
      const segFromPrev = index > 0 ? segs[index - 1] : null;
      const lat = coords ? coords.lat : (det.lat !== undefined ? det.lat : 0);
      const lng = coords ? coords.lng : (det.lng !== undefined ? det.lng : 0);
      const camName = det.cameraName || det.cameraLocation || `CAM-${det.cameraId}`;
      const camLoc = det.cameraLocation || det.location || '';

      return {
        id: det.id,
        plate: plateNorm,
        cameraId: det.cameraId,
        cameraName: camName,
        cameraLocation: camLoc,
        location: det.location || camLoc,
        zone: det.zone || '',
        timestamp: det.timestamp,
        latitude: lat,
        longitude: lng,
        lat,
        lng,
        confidence: det.confidence !== undefined ? det.confidence : 0.9,
        ocrConfidence: det.ocrConfidence !== undefined ? det.ocrConfidence : 0.9,
        detectorConfidence: det.detectorConfidence !== undefined ? det.detectorConfidence : 0.9,
        vehicleType: det.vehicleType || 'car',
        direction: det.direction || 'unknown',
        simulated: !!det.simulated,
        imagePath: det.imagePath || null,
        distanceFromPreviousKm: segFromPrev ? segFromPrev.distanceKm : null,
        travelTimeSeconds: segFromPrev ? segFromPrev.travelTimeSeconds : null,
        travelMins: segFromPrev ? segFromPrev.travelMins : null,
        estimatedSpeedKmh: segFromPrev ? segFromPrev.estimatedSpeedKmh : null,
        validForAnalytics: segFromPrev ? segFromPrev.validForAnalytics : true,
        validationIssues: segFromPrev ? segFromPrev.validationIssues : [],
        anomaly: segFromPrev ? segFromPrev.anomaly : null,
      };
    });

    const cameras = [...new Set(sessionPoints.map((p) => p.cameraName || `CAM-${p.cameraId}`))];

    return {
      journeyId,
      sessionIndex: idx,
      sessionStart: startTime,
      sessionEnd: endTime,
      sessionSightings: sGroup,
      totalSightings: sGroup.length,
      durationSeconds,
      durationMinutes,
      segments: segs,
      points: sessionPoints,
      cameras,
      geodesicDistanceKm,
      roadDistanceKm,
      totalDistanceKm: summary.totalDistanceKm, // Preserve backward compatibility for assertions
      averageJourneySpeedKmh: summary.averageJourneySpeedKmh,
      roadAligned: roadRoute ? roadRoute.roadAligned : false,
      routingStatus: roadRoute ? roadRoute.routingStatus : 'UNAVAILABLE',
      roadGeometry: roadRoute ? roadRoute.geometry : null,
      roadLatLngs: roadRoute ? roadRoute.latLngs : [],
      roadLegs: roadRoute ? roadRoute.legs : [],
      isActive: idx === rawGroups.length - 1,
      splitReason: group.splitReason,
    };
  });
}

/**
 * Builds the complete trajectory payload for a given registration plate.
 * Integrates journey sessionization, segment-by-segment analytics, road-network routing,
 * and next-camera route prediction.
 * Fully backward-compatible with existing UI and test suites.
 *
 * @param {string} plate
 * @param {Array<object>} detections
 * @param {object} [options]
 * @returns {object}
 */
function buildTrajectory(plate, detections, options = {}) {
  const normalized = normalizePlate(plate);
  if (!normalized) {
    return {
      success: false,
      found: false,
      plate: '',
      message: 'Plate number is required',
      trail: [],
      points: [],
      segments: [],
      totalSightings: 0,
    };
  }

  const sightings = getPlateSightings(normalized, detections);

  if (sightings.length === 0) {
    return {
      success: false,
      found: false,
      plate: normalized,
      message: 'No detections found for this plate',
      trail: [],
      points: [],
      segments: [],
      totalSightings: 0,
      totalDistanceKm: 0,
      totalTravelTimeSeconds: 0,
      totalTravelTimeMinutes: 0,
      totalMinutes: 0,
      averageJourneySpeedKmh: null,
      validSegmentCount: 0,
      invalidSegmentCount: 0,
      cameras: [],
      firstSeen: null,
      lastSeen: null,
      journeys: [],
      activeJourney: null,
      predictions: [],
    };
  }

  const config = options.config || loadAnalyticsConfig();
  const journeys = segmentSightingsIntoJourneys(sightings, config);

  // Identify target journey session (default to most recent / active journey)
  let activeJourney = journeys[journeys.length - 1];
  if (options.journeyId) {
    const found = journeys.find((j) => j.journeyId === options.journeyId);
    if (found) activeJourney = found;
  }

  // Also calculate full sightings segments for complete history / anomaly checks
  const allSegments = calculateTrajectorySegments(sightings, config);
  const allSummary = calculateTrajectorySummary(sightings, allSegments);

  // Phase 18 Deterministic Trajectory Route Anomalies across active journey (and all)
  const routeAnomalies = [];
  const evalSegments = activeJourney ? activeJourney.segments : allSegments;
  const evalSightings = activeJourney ? activeJourney.sessionSightings : sightings;

  evalSegments.forEach((seg) => {
    if (seg.anomaly && seg.anomaly.type === 'IMPOSSIBLE_TRAVEL') {
      routeAnomalies.push({
        type: 'IMPOSSIBLE_TRAVEL',
        severity: 'HIGH',
        description: seg.anomaly.reason,
        fromCamera: seg.fromCameraName,
        toCamera: seg.toCameraName,
        timestamp: seg.toTimestamp,
        speedKmh: seg.anomaly.calculatedSpeedKmh,
      });
    }
  });

  const camVisitCounts = {};
  evalSightings.forEach((s) => {
    const cId = String(s.cameraName || s.cameraLocation || s.cameraId || 'UNKNOWN');
    camVisitCounts[cId] = (camVisitCounts[cId] || 0) + 1;
  });
  Object.entries(camVisitCounts).forEach(([cId, count]) => {
    if (count >= 3) {
      routeAnomalies.push({
        type: 'REPEATED_LOOP',
        severity: 'MEDIUM',
        description: `Vehicle visited camera ${cId} ${count} times during trajectory, indicating circling or loitering pattern.`,
        camera: cId,
        occurrences: count,
      });
    }
  });

  evalSightings.forEach((s) => {
    const zoneStr = String(s.zone || s.cameraZone || '').toUpperCase();
    if (zoneStr.includes('RESTRICTED') || zoneStr.includes('RED') || s.isRestrictedZone) {
      routeAnomalies.push({
        type: 'RESTRICTED_ZONE_ENTRY',
        severity: 'HIGH',
        description: `Vehicle detected entering restricted zone: ${s.zone || 'RESTRICTED'} at camera ${s.cameraName || s.cameraId}`,
        cameraId: s.cameraId,
        zone: s.zone,
        timestamp: s.timestamp,
      });
    }
  });

  evalSegments.forEach((seg) => {
    if (
      !seg.isSameCamera &&
      seg.travelTimeSeconds !== null &&
      seg.travelTimeSeconds > 0 &&
      seg.travelTimeSeconds < 5
    ) {
      routeAnomalies.push({
        type: 'UNUSUAL_RAPID_SEQUENCE',
        severity: 'MEDIUM',
        description: `Rapid inter-camera transition detected (${seg.travelTimeSeconds}s between ${seg.fromCameraName} and ${seg.toCameraName}).`,
        fromCamera: seg.fromCameraName,
        toCamera: seg.toCameraName,
        travelTimeSeconds: seg.travelTimeSeconds,
      });
    }
  });

  // Calculate Next-Camera Predictions based on ACTIVE journey using fleet-wide historical journey corpus
  const fleetJourneys = Array.isArray(detections) && detections.length > 5
    ? extractFleetJourneys(detections, config)
    : journeys;
  const predictions = predictNextCamerasSync(activeJourney, fleetJourneys);

  // Return comprehensive trajectory payload
  return {
    success: true,
    found: true,
    plate: normalized,
    journeyId: activeJourney.journeyId,
    sessionStart: activeJourney.sessionStart,
    sessionEnd: activeJourney.sessionEnd,
    totalSightings: activeJourney.totalSightings,
    totalDistanceKm: activeJourney.totalDistanceKm, // Geodesic for backward compatibility
    roadDistanceKm: activeJourney.roadDistanceKm, // Road distance when road routing succeeds
    geodesicDistanceKm: activeJourney.geodesicDistanceKm,
    totalTravelTimeSeconds: activeJourney.durationSeconds,
    totalTravelTimeMinutes: activeJourney.durationMinutes,
    totalMinutes: Math.round(activeJourney.durationMinutes),
    averageJourneySpeedKmh: activeJourney.averageJourneySpeedKmh,
    validSegmentCount: activeJourney.segments.filter((s) => s.validForAnalytics).length,
    invalidSegmentCount: activeJourney.segments.filter((s) => !s.validForAnalytics).length,
    firstSeen: activeJourney.sessionStart,
    lastSeen: activeJourney.sessionEnd,
    cameras: activeJourney.cameras,
    trail: activeJourney.points,
    points: activeJourney.points,
    segments: activeJourney.segments,
    roadGeometry: activeJourney.roadGeometry,
    roadLatLngs: activeJourney.roadLatLngs,
    roadAligned: activeJourney.roadAligned,
    routingStatus: activeJourney.routingStatus,
    anomalies: routeAnomalies,
    hasAnomalies: routeAnomalies.length > 0,
    activeJourney,
    journeys: journeys.map((j) => ({
      journeyId: j.journeyId,
      sessionIndex: j.sessionIndex,
      sessionStart: j.sessionStart,
      sessionEnd: j.sessionEnd,
      sightingsCount: j.totalSightings,
      distanceKm: j.totalDistanceKm,
      roadDistanceKm: j.roadDistanceKm,
      durationMinutes: j.durationMinutes,
      averageJourneySpeedKmh: j.averageJourneySpeedKmh,
      roadAligned: j.roadAligned,
      isActive: j.isActive,
      splitReason: j.splitReason,
    })),
    allJourneysCount: journeys.length,
    allSightingsCount: sightings.length,
    predictions: predictions.nextCameras || [],
    multiHopPrediction: predictions.multiHopRoute || null,
  };
}

/**
 * Async version of buildTrajectory that can query external OSRM endpoints asynchronously.
 */
async function buildTrajectoryAsync(plate, detections, options = {}) {
  const syncResult = buildTrajectory(plate, detections, options);
  if (!syncResult.found || !syncResult.activeJourney) {
    return syncResult;
  }

  // If road geometry was not already aligned via pre-cached topology, fetch async
  if (!syncResult.roadAligned && syncResult.activeJourney.totalSightings >= 2) {
    const camSequence = syncResult.activeJourney.sessionSightings.map((s) => s.cameraId);
    const roadRoute = await getRoadRouteThroughCameras(camSequence);
    if (roadRoute.roadAligned) {
      syncResult.roadGeometry = roadRoute.geometry;
      syncResult.roadLatLngs = roadRoute.latLngs;
      syncResult.roadDistanceKm = roadRoute.totalDistanceKm;
      syncResult.roadAligned = true;
      syncResult.routingStatus = 'OK';
      syncResult.activeJourney.roadGeometry = roadRoute.geometry;
      syncResult.activeJourney.roadLatLngs = roadRoute.latLngs;
      syncResult.activeJourney.roadDistanceKm = roadRoute.totalDistanceKm;
      syncResult.activeJourney.roadAligned = true;
    }
  }

  // Also query async predictions if needed
  if (!syncResult.predictions || syncResult.predictions.length === 0) {
    const asyncPred = await predictNextCameras(syncResult.activeJourney, syncResult.journeys || []);
    syncResult.predictions = asyncPred.nextCameras || [];
    syncResult.multiHopPrediction = asyncPred.multiHopRoute || null;
  }

  return syncResult;
}

/**
 * Extracts all multi-camera journey sessions across all vehicles in the detections corpus.
 */
function extractFleetJourneys(allDetections, config = null) {
  if (!Array.isArray(allDetections) || allDetections.length === 0) return [];
  const plates = [...new Set(allDetections.map((d) => normalizePlate(d.plate)))];
  const fleetJourneys = [];
  plates.forEach((p) => {
    const s = getPlateSightings(p, allDetections);
    const jList = segmentSightingsIntoJourneys(s, config);
    jList.forEach((j) => {
      if ((j.sessionSightings || j.sightings || []).length >= 2) {
        fleetJourneys.push(j);
      }
    });
  });
  return fleetJourneys;
}

module.exports = {
  loadAnalyticsConfig,
  normalizePlate,
  getCoordinates,
  getPlateSightings,
  segmentSightingsIntoJourneys,
  extractFleetJourneys,
  calculateTrajectorySegments,
  calculateTrajectorySummary,
  buildTrajectory,
  buildTrajectoryAsync,
};

