'use strict';

/**
 * Central Analytics Service Coordinator & Summary API
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const {
  loadCameras,
  formatStandardCameraId,
  filterDetectionsByTime,
  deduplicateDetections,
} = require('./analyticsUtils');
const trafficAnalytics = require('./trafficAnalytics');
const routeAnalytics = require('./routeAnalytics');
const speedAnalytics = require('./speedAnalytics');
const odAnalytics = require('./odAnalytics');
const congestionAnalytics = require('./congestionAnalytics');
const trendAnalytics = require('./trendAnalytics');
const trajectoryService = require('./trajectoryService');

// In-memory short-lived cache (5 seconds) to handle rapid dashboard polling
const cache = {
  summary: { timestamp: 0, data: null, detectionsLength: 0 },
};

/**
 * Generates high-level analytics summary metrics for dashboard cards and system status.
 *
 * @param {Array<object>} detections
 * @param {Array<object>} [alertHistory]
 * @param {object} [camerasStatus]
 * @returns {object}
 */
function getSummary(detections, alertHistory = [], camerasStatus = {}) {
  const now = Date.now();

  // Check cache validity (valid for 4 seconds if detection reference and count has not changed)
  if (
    cache.summary.data &&
    cache.summary.lastDetectionsRef === detections &&
    now - cache.summary.timestamp < 4000 &&
    cache.summary.detectionsLength === detections.length
  ) {
    return cache.summary.data;
  }

  // Determine reference time for "last hour"
  // In real-time production, refTime is Date.now().
  // If analyzing historical / test datasets where all detections occurred > 2 hours ago,
  // anchor the 1-hour window to the latest detection timestamp in the dataset.
  let refTime = now;
  if (detections.length > 0) {
    let maxTs = 0;
    for (const d of detections) {
      if (d.timestamp) {
        const t = new Date(d.timestamp).getTime();
        if (!isNaN(t) && t > maxTs) {
          maxTs = t;
        }
      }
    }
    if (maxTs > 0 && (now - maxTs > 2 * 3600 * 1000)) {
      refTime = maxTs;
    }
  }

  const oneHourAgo = new Date(refTime - 3600 * 1000);
  const toDate = new Date(refTime);
  const lastHourDetections = filterDetectionsByTime(detections, oneHourAgo, toDate);
  const dedupedLastHour = deduplicateDetections(lastHourDetections, 10);

  const cameras = loadCameras();

  // Active cameras: either reported connected via WebRTC or detected a vehicle in the last hour
  const connectedCameraIds = new Set(
    Object.entries(camerasStatus || {})
      .filter(([_, status]) => status === 'connected')
      .map(([id]) => formatStandardCameraId(id))
  );

  const camerasWithSightings = new Set(
    lastHourDetections.map((d) => formatStandardCameraId(d.cameraId))
  );

  const activeCameraCount = Math.max(
    connectedCameraIds.size,
    camerasWithSightings.size,
    cameras.length > 0 ? cameras.filter((c) => camerasWithSightings.has(c.standardId)).length : 0
  );

  // Unique vehicles in the last hour
  const uniquePlatesSet = new Set(
    dedupedLastHour
      .map((d) => (d.plate || '').toString().toUpperCase().replace(/[^A-Z0-9]/g, ''))
      .filter(Boolean)
  );

  // Speed and congestion in the last hour
  const speedRes = speedAnalytics.calculateSpeedAnalytics(lastHourDetections);
  const congRes = congestionAnalytics.calculateCongestion(lastHourDetections, detections);
  const congestedRoutesCount = congRes.filter((r) => r.level === 'HIGH' || r.level === 'SEVERE').length;

  // Diagnostics: count invalid trajectory segments across last hour
  let invalidSegmentsCount = 0;
  const plateMap = new Map();
  lastHourDetections.forEach((d) => {
    const p = trajectoryService.normalizePlate(d.plate);
    if (!p) return;
    if (!plateMap.has(p)) plateMap.set(p, []);
    plateMap.get(p).push(d);
  });

  plateMap.forEach((dets) => {
    if (dets.length >= 2) {
      const segs = trajectoryService.calculateTrajectorySegments(dets);
      invalidSegmentsCount += segs.filter((s) => !s.validForAnalytics).length;
    }
  });

  const summary = {
    success: true,
    timestamp: new Date().toISOString(),
    activeCameras: activeCameraCount || cameras.length,
    vehiclesLastHour: dedupedLastHour.length,
    uniqueVehiclesLastHour: uniquePlatesSet.size,
    totalDetectionsAllTime: detections.length,
    estimatedAverageSpeedKmh: speedRes.overallEstimatedAverageSpeedKmh,
    congestedRoutes: congestedRoutesCount,
    watchlistAlerts: Array.isArray(alertHistory) ? alertHistory.length : 0,
    invalidTrajectorySegments: invalidSegmentsCount,
  };

  cache.summary = {
    timestamp: now,
    data: summary,
    detectionsLength: detections.length,
    lastDetectionsRef: detections,
  };

  return summary;
}

module.exports = {
  getSummary,
  trafficAnalytics,
  routeAnalytics,
  speedAnalytics,
  odAnalytics,
  congestionAnalytics,
  trendAnalytics,
};
