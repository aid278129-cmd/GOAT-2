'use strict';

/**
 * Origin-Destination (OD) Analytics Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const {
  loadCameras,
  formatStandardCameraId,
  resolveCamera,
  parseTimeRange,
  filterDetectionsByTime,
} = require('./analyticsUtils');
const trajectoryService = require('./trajectoryService');

/**
 * Performs Origin-Destination analysis across unique vehicle journeys within the selected time window.
 *
 * @param {Array<object>} detections
 * @param {object} [options]
 * @returns {object}
 */
function calculateOriginDestination(detections, options = {}) {
  const timeFiltered = options.timeRange && (options.timeRange.fromDate || options.timeRange.toDate)
    ? filterDetectionsByTime(detections, options.timeRange.fromDate, options.timeRange.toDate)
    : detections;

  const cameras = loadCameras();
  const allowSameOriginDestination = !!options.allowSameOriginDestination;

  // Group detections by plate
  const plateMap = new Map();
  timeFiltered.forEach((d) => {
    const plate = trajectoryService.normalizePlate(d.plate);
    if (!plate) return;
    if (!plateMap.has(plate)) {
      plateMap.set(plate, []);
    }
    plateMap.get(plate).push(d);
  });

  const pairCounts = new Map(); // key `${orig}->${dest}`
  let totalJourneysEvaluated = 0;

  plateMap.forEach((dets, plate) => {
    if (dets.length < 2) return;

    // Chronological sorting
    const sorted = [...dets].sort((a, b) => {
      const tA = new Date(a.timestamp).getTime();
      const tB = new Date(b.timestamp).getTime();
      return (isNaN(tA) ? 0 : tA) - (isNaN(tB) ? 0 : tB);
    });

    const originCam = resolveCamera(sorted[0].cameraId, cameras);
    const destCam = resolveCamera(sorted[sorted.length - 1].cameraId, cameras);

    const origId = originCam.standardId;
    const destId = destCam.standardId;

    if (!allowSameOriginDestination && origId === destId) {
      return; // Skip non-journey stationary observations
    }

    const key = `${origId}->${destId}`;
    if (!pairCounts.has(key)) {
      pairCounts.set(key, {
        originCameraId: origId,
        destinationCameraId: destId,
        originCameraNumericId: originCam.cameraId,
        destinationCameraNumericId: destCam.cameraId,
        originCameraName: originCam.name,
        destinationCameraName: destCam.name,
        originLocation: originCam.roadName,
        destinationLocation: destCam.roadName,
        vehicleCount: 0,
      });
    }

    pairCounts.get(key).vehicleCount++;
    totalJourneysEvaluated++;
  });

  // Convert to sorted array
  const pairs = Array.from(pairCounts.values()).sort((a, b) => b.vehicleCount - a.vehicleCount);

  // Build Square Matrix
  const cameraIds = cameras.map((c) => c.standardId);
  const cameraNames = cameras.map((c) => c.name);

  const matrix = cameraIds.map((origId) => {
    return cameraIds.map((destId) => {
      const key = `${origId}->${destId}`;
      const entry = pairCounts.get(key);
      return entry ? entry.vehicleCount : 0;
    });
  });

  const matrixObject = {
    cameras: cameraIds,
    cameraNames,
    grid: matrix,
    matrix,
  };

  return {
    totalJourneysEvaluated,
    uniqueVehiclesEvaluated: plateMap.size,
    pairs,
    odPairs: pairs,
    cameras: cameraIds,
    cameraNames,
    matrix: matrixObject,
    matrixGrid: matrix,
  };
}

/**
 * Main handler function for GET /api/analytics/origin-destination
 */
function getOriginDestinationAnalytics(detections, query = {}) {
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  const result = calculateOriginDestination(detections, {
    timeRange: timeRes,
    allowSameOriginDestination: query.includeSame === 'true',
  });

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    ...result,
  };
}

module.exports = {
  calculateOriginDestination,
  getOriginDestinationAnalytics,
};
