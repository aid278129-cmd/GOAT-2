'use strict';

/**
 * Trajectory Reconstruction and Analysis Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const fs = require('fs');
const path = require('path');
const { haversineDistance, calculateEstimatedSpeed } = require('./geoService');

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
 * Builds the complete trajectory payload for a given registration plate.
 * Integrates sightings, segment-by-segment analytics, and summary statistics.
 * Fully backward-compatible with the existing UI and endpoints.
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
    };
  }

  const config = options.config || loadAnalyticsConfig();
  const segments = calculateTrajectorySegments(sightings, config);
  const summary = calculateTrajectorySummary(sightings, segments);

  // Build enhanced points array preserving legacy keys for dashboard & map
  const points = sightings.map((det, index) => {
    const coords = getCoordinates(det);
    const segFromPrev = index > 0 ? segments[index - 1] : null;

    const lat = coords ? coords.lat : (det.lat !== undefined ? det.lat : 0);
    const lng = coords ? coords.lng : (det.lng !== undefined ? det.lng : 0);

    const camName = det.cameraName || det.cameraLocation || `CAM-${det.cameraId}`;
    const camLoc = det.cameraLocation || det.location || '';

    return {
      id: det.id,
      plate: normalized,
      cameraId: det.cameraId,
      cameraName: camName,
      cameraLocation: camLoc,
      location: det.location || camLoc, // legacy compatibility
      zone: det.zone || '',
      timestamp: det.timestamp,
      latitude: lat,
      longitude: lng,
      lat, // legacy compatibility for Leaflet
      lng, // legacy compatibility for Leaflet
      confidence: det.confidence !== undefined ? det.confidence : 0.9,
      ocrConfidence: det.ocrConfidence !== undefined ? det.ocrConfidence : 0.9,
      detectorConfidence: det.detectorConfidence !== undefined ? det.detectorConfidence : 0.9,
      vehicleType: det.vehicleType || 'car',
      direction: det.direction || 'unknown',
      simulated: !!det.simulated,
      imagePath: det.imagePath || null,

      // Segment metadata relative to previous sighting
      distanceFromPreviousKm: segFromPrev ? segFromPrev.distanceKm : null,
      travelTimeSeconds: segFromPrev ? segFromPrev.travelTimeSeconds : null,
      travelMins: segFromPrev ? segFromPrev.travelMins : null,
      estimatedSpeedKmh: segFromPrev ? segFromPrev.estimatedSpeedKmh : null,
      validForAnalytics: segFromPrev ? segFromPrev.validForAnalytics : true,
      validationIssues: segFromPrev ? segFromPrev.validationIssues : [],
      anomaly: segFromPrev ? segFromPrev.anomaly : null,
    };
  });

  const uniqueCameras = [
    ...new Set(points.map((p) => p.cameraName || `CAM-${p.cameraId}`)),
  ];

  // Phase 18: Deterministic Trajectory Route Anomalies
  const routeAnomalies = [];

  // Rule 1: IMPOSSIBLE_TRAVEL from segments
  segments.forEach((seg) => {
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

  // Rule 2: REPEATED_LOOP (Camera visited 3+ times in trajectory)
  const camVisitCounts = {};
  sightings.forEach((s) => {
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

  // Rule 3: RESTRICTED_ZONE_ENTRY
  sightings.forEach((s) => {
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

  // Rule 4: UNUSUAL_RAPID_SEQUENCE (< 5s between distinct cameras)
  segments.forEach((seg) => {
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

  return {
    success: true,
    found: true,
    plate: normalized,
    totalSightings: summary.totalSightings,
    totalDistanceKm: summary.totalDistanceKm,
    totalTravelTimeSeconds: summary.totalTravelTimeSeconds,
    totalTravelTimeMinutes: summary.totalTravelTimeMinutes,
    totalMinutes: Math.round(summary.totalTravelTimeMinutes), // legacy compatibility
    averageJourneySpeedKmh: summary.averageJourneySpeedKmh,
    validSegmentCount: summary.validSegmentCount,
    invalidSegmentCount: summary.invalidSegmentCount,
    firstSeen: summary.firstSeen,
    lastSeen: summary.lastSeen,
    cameras: uniqueCameras,
    trail: points, // legacy compatibility for dashboard & tracking
    points, // roadmap standard
    segments, // enhanced segments with distance, speed, anomalies
    anomalies: routeAnomalies, // Phase 18 deterministic route anomalies
    hasAnomalies: routeAnomalies.length > 0,
  };
}

module.exports = {
  loadAnalyticsConfig,
  normalizePlate,
  getCoordinates,
  getPlateSightings,
  calculateTrajectorySegments,
  calculateTrajectorySummary,
  buildTrajectory,
};
