'use strict';

/**
 * Common Analytics Utilities
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const fs = require('fs');
const path = require('path');

const CONFIG_FILE = path.join(__dirname, '..', 'data', 'analytics_config.json');
const CAMERAS_FILE = path.join(__dirname, '..', 'data', 'cameras.json');

/**
 * Loads analytics configuration with fallbacks.
 */
function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    }
  } catch (err) {
    console.warn('⚠️  Could not load analytics_config.json:', err.message);
  }
  return {
    maxReasonableSpeedKmh: 180,
    minSegmentTravelTimeSeconds: 1,
    routeLoopThreshold: 3,
    routeLoopWindowMinutes: 60,
    duplicateDetectionWindowSeconds: 10,
    defaultAnalyticsIntervalMinutes: 15,
    densityThresholds: {
      lowMax: 100,
      mediumMax: 250,
      highMax: 500,
    },
    congestion: {
      densityMultiplierHigh: 1.5,
      speedRatioHigh: 0.6,
      minimumVehiclesForEvaluation: 5,
      defaultBaselineSpeedKmh: 40.0,
      defaultBaselineVolumePerHour: 250,
    },
  };
}

/**
 * Loads camera metadata list from data/cameras.json with fallback defaults.
 */
function loadCameras() {
  try {
    if (fs.existsSync(CAMERAS_FILE)) {
      const data = JSON.parse(fs.readFileSync(CAMERAS_FILE, 'utf8'));
      if (Array.isArray(data) && data.length > 0) {
        return data.map((c) => ({
          cameraId: c.cameraId,
          standardId: formatStandardCameraId(c.cameraId),
          name: c.name || `CAM-${c.cameraId}`,
          latitude: Number(c.latitude || c.lat || 0),
          longitude: Number(c.longitude || c.lng || 0),
          roadName: c.roadName || c.location || '',
          direction: c.direction || 'unknown',
          zone: c.zone || 'Central',
        }));
      }
    }
  } catch (err) {
    console.warn('⚠️  Could not load cameras.json:', err.message);
  }

  return [
    { cameraId: 1, standardId: 'CAM_01', name: 'Junction A', latitude: 13.0827, longitude: 80.2707, roadName: 'Chennai Central', direction: 'northbound', zone: 'North' },
    { cameraId: 2, standardId: 'CAM_02', name: 'Junction B', latitude: 13.0731, longitude: 80.2609, roadName: 'T. Nagar', direction: 'eastbound', zone: 'West' },
    { cameraId: 3, standardId: 'CAM_03', name: 'Junction C', latitude: 13.0878, longitude: 80.2785, roadName: 'Perambur', direction: 'southbound', zone: 'North-East' },
    { cameraId: 4, standardId: 'CAM_04', name: 'Highway Entry', latitude: 13.0569, longitude: 80.2425, roadName: 'Guindy', direction: 'westbound', zone: 'South' },
  ];
}

/**
 * Formats a numeric or string camera ID to canonical CAM_01 format.
 */
function formatStandardCameraId(id) {
  if (id === null || id === undefined) return 'CAM_00';
  const digits = String(id).replace(/\D/g, '');
  if (!digits) {
    const clean = String(id).trim().toUpperCase();
    return clean.startsWith('CAM_') ? clean : `CAM_${clean}`;
  }
  const num = parseInt(digits, 10);
  return `CAM_${num < 10 ? '0' + num : num}`;
}

/**
 * Resolves any historical camera ID (1, "1", "CAM_01", "CAM-1", etc.)
 * to its standard ID and camera configuration record.
 */
function resolveCamera(id, camerasList = null) {
  const list = camerasList || loadCameras();
  if (id === null || id === undefined) {
    return {
      cameraId: 0,
      standardId: 'CAM_00',
      name: 'Unknown Camera',
      latitude: 0,
      longitude: 0,
      roadName: '',
      direction: 'unknown',
      zone: 'Unknown',
    };
  }

  const str = String(id).trim().toUpperCase();
  const digits = str.replace(/\D/g, '');
  const num = digits ? parseInt(digits, 10) : null;
  const standardId = formatStandardCameraId(id);

  const matched = list.find((c) => {
    return (
      c.standardId === standardId ||
      String(c.cameraId) === str ||
      (num !== null && c.cameraId === num) ||
      c.name.toUpperCase() === str
    );
  });

  if (matched) return matched;

  return {
    cameraId: num || id,
    standardId,
    name: str.startsWith('CAM') ? str : `CAM-${id}`,
    latitude: 0,
    longitude: 0,
    roadName: '',
    direction: 'unknown',
    zone: 'Unknown',
  };
}

/**
 * Validates and parses time range query parameters.
 * Returns { valid: true, fromDate, toDate, fromIso, toIso } or { valid: false, message }.
 */
function parseTimeRange(fromStr, toStr) {
  let fromDate = null;
  let toDate = null;

  if (fromStr) {
    const parsedFrom = new Date(fromStr);
    if (isNaN(parsedFrom.getTime())) {
      return { valid: false, message: 'Invalid query parameter "from": must be a valid ISO timestamp' };
    }
    fromDate = parsedFrom;
  }

  if (toStr) {
    const parsedTo = new Date(toStr);
    if (isNaN(parsedTo.getTime())) {
      return { valid: false, message: 'Invalid query parameter "to": must be a valid ISO timestamp' };
    }
    toDate = parsedTo;
  }

  if (fromDate && toDate && fromDate.getTime() > toDate.getTime()) {
    return { valid: false, message: 'Invalid time range: "from" timestamp must be earlier than or equal to "to" timestamp' };
  }

  return {
    valid: true,
    fromDate,
    toDate,
    fromIso: fromDate ? fromDate.toISOString() : null,
    toIso: toDate ? toDate.toISOString() : null,
  };
}

/**
 * Validates and parses interval query string ('5m', '15m', '30m', '1h').
 */
function parseInterval(intervalStr, defaultInterval = '15m') {
  const chosen = (intervalStr || defaultInterval).toLowerCase().trim();
  const intervals = {
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '60m': 60 * 60 * 1000,
  };

  if (!intervals[chosen]) {
    return {
      valid: false,
      message: 'Invalid interval parameter. Allowed intervals are: 5m, 15m, 30m, 1h',
    };
  }

  return {
    valid: true,
    interval: chosen === '60m' ? '1h' : chosen,
    durationMs: intervals[chosen],
  };
}

/**
 * Filters detections by a given time range safely without mutating source.
 */
function filterDetectionsByTime(detections, fromDate, toDate) {
  if (!Array.isArray(detections)) return [];
  const fromMs = fromDate ? fromDate.getTime() : -Infinity;
  const toMs = toDate ? toDate.getTime() : Infinity;

  return detections.filter((d) => {
    if (!d || !d.timestamp) return false;
    const t = new Date(d.timestamp).getTime();
    if (isNaN(t)) return false;
    return t >= fromMs && t <= toMs;
  });
}

/**
 * Deduplicates detections so repeated frames of the same plate at the same camera
 * within `duplicateDetectionWindowSeconds` count as a single camera sighting.
 */
function deduplicateDetections(detections, windowSeconds = 10) {
  if (!Array.isArray(detections) || detections.length === 0) return [];
  const windowMs = (Number(windowSeconds) || 10) * 1000;

  // Shallow clone and sort chronologically
  const sorted = [...detections].sort((a, b) => {
    const tA = new Date(a.timestamp).getTime();
    const tB = new Date(b.timestamp).getTime();
    return (isNaN(tA) ? 0 : tA) - (isNaN(tB) ? 0 : tB);
  });

  const lastSeenMap = new Map(); // key `${plate}:${standardCamId}` -> lastSeenTimestampMs
  const deduplicated = [];

  for (const det of sorted) {
    const plate = (det.plate || '').toString().toUpperCase().replace(/[^A-Z0-9]/g, '');
    const standardCamId = formatStandardCameraId(det.cameraId);
    const t = new Date(det.timestamp).getTime();

    if (!plate || isNaN(t)) {
      continue; // Skip corrupted records
    }

    const key = `${plate}:${standardCamId}`;
    const lastTime = lastSeenMap.get(key);

    if (lastTime !== undefined && t - lastTime < windowMs) {
      // Within duplicate window — skip counting as independent vehicle entry
      continue;
    }

    lastSeenMap.set(key, t);
    deduplicated.push(det);
  }

  return deduplicated;
}

module.exports = {
  loadConfig,
  loadCameras,
  formatStandardCameraId,
  resolveCamera,
  parseTimeRange,
  parseInterval,
  filterDetectionsByTime,
  deduplicateDetections,
};
