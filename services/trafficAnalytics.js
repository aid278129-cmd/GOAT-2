'use strict';

/**
 * Traffic Density Analytics Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const {
  loadConfig,
  loadCameras,
  formatStandardCameraId,
  resolveCamera,
  parseTimeRange,
  parseInterval,
  filterDetectionsByTime,
  deduplicateDetections,
} = require('./analyticsUtils');

/**
 * Classifies traffic density level according to configurable thresholds.
 */
function classifyDensityLevel(count, thresholds = null) {
  const cfg = thresholds || (loadConfig().densityThresholds || {});
  const lowMax = Number(cfg.lowMax) || 100;
  const mediumMax = Number(cfg.mediumMax) || 250;
  const highMax = Number(cfg.highMax) || 500;

  if (count <= lowMax) return 'LOW';
  if (count <= mediumMax) return 'MEDIUM';
  if (count <= highMax) return 'HIGH';
  return 'SEVERE';
}

/**
 * Calculates vehicle counts and unique plates per camera node.
 */
function getVehicleCountByCamera(detections, timeRange = null, config = null) {
  const cfg = config || loadConfig();
  const timeFiltered = timeRange && (timeRange.fromDate || timeRange.toDate)
    ? filterDetectionsByTime(detections, timeRange.fromDate, timeRange.toDate)
    : detections;

  const deduped = deduplicateDetections(timeFiltered, cfg.duplicateDetectionWindowSeconds);
  const cameras = loadCameras();

  const countMap = new Map();
  const rawCountMap = new Map();
  const platesMap = new Map();

  cameras.forEach((c) => {
    countMap.set(c.standardId, 0);
    rawCountMap.set(c.standardId, 0);
    platesMap.set(c.standardId, new Set());
  });

  timeFiltered.forEach((det) => {
    const cam = resolveCamera(det.cameraId, cameras);
    const standardId = cam.standardId;
    if (!rawCountMap.has(standardId)) {
      rawCountMap.set(standardId, 0);
    }
    rawCountMap.set(standardId, rawCountMap.get(standardId) + 1);
  });

  deduped.forEach((det) => {
    const cam = resolveCamera(det.cameraId, cameras);
    const standardId = cam.standardId;

    if (!countMap.has(standardId)) {
      countMap.set(standardId, 0);
      platesMap.set(standardId, new Set());
    }

    countMap.set(standardId, countMap.get(standardId) + 1);

    const plate = (det.plate || '').toString().toUpperCase().replace(/[^A-Z0-9]/g, '');
    if (plate) {
      platesMap.get(standardId).add(plate);
    }
  });

  return { countMap, rawCountMap, platesMap, dedupedTotal: deduped.length, rawTotal: timeFiltered.length };
}

/**
 * Returns traffic density metrics for each camera node.
 */
function getDensityByCamera(detections, options = {}) {
  const cfg = options.config || loadConfig();
  const timeRange = options.timeRange || { fromDate: null, toDate: null };
  const interval = options.interval || `${cfg.defaultAnalyticsIntervalMinutes || 15}m`;
  const filterCameraId = options.cameraId ? formatStandardCameraId(options.cameraId) : null;

  const { countMap, rawCountMap, platesMap } = getVehicleCountByCamera(detections, timeRange, cfg);
  const cameras = loadCameras();

  let durationMinutes = 15;
  if (timeRange.fromDate && timeRange.toDate) {
    const diffMs = timeRange.toDate.getTime() - timeRange.fromDate.getTime();
    if (diffMs > 0) durationMinutes = diffMs / 60000;
  } else if (interval) {
    const intRes = parseInterval(interval);
    if (intRes.valid) durationMinutes = intRes.durationMs / 60000;
  }

  const results = cameras
    .filter((c) => !filterCameraId || c.standardId === filterCameraId || String(c.cameraId) === filterCameraId)
    .map((c) => {
      const totalDets = rawCountMap ? (rawCountMap.get(c.standardId) || 0) : 0;
      const vehicleCount = countMap.get(c.standardId) || 0;
      const uniquePlates = platesMap.get(c.standardId) ? platesMap.get(c.standardId).size : 0;
      const densityLevel = classifyDensityLevel(vehicleCount, cfg.densityThresholds);

      const flowRatePerMinute = durationMinutes > 0 ? Number((vehicleCount / durationMinutes).toFixed(2)) : 0;
      const flowRatePer5Min = durationMinutes > 0 ? Number(((vehicleCount / durationMinutes) * 5).toFixed(2)) : 0;
      const flowRatePerHour = durationMinutes > 0 ? Number(((vehicleCount / durationMinutes) * 60).toFixed(1)) : 0;

      return {
        cameraId: c.standardId,
        cameraNumericId: c.cameraId,
        cameraName: c.name,
        location: c.roadName,
        roadName: c.roadName,
        zone: c.zone,
        latitude: c.latitude,
        longitude: c.longitude,
        totalDetections: totalDets,
        vehicleCount,
        uniquePlateCount: uniquePlates,
        uniqueVehicles: uniquePlates,
        flowRatePerMinute,
        flowRatePer5Min,
        flowRatePerHour,
        interval,
        densityLevel,
      };
    });

  return results;
}

/**
 * Groups and returns traffic density aggregated by zone.
 */
function getDensityByZone(detections, options = {}) {
  const cameraDensities = getDensityByCamera(detections, options);
  const zoneMap = new Map();

  cameraDensities.forEach((cd) => {
    const z = cd.zone || 'Central';
    if (!zoneMap.has(z)) {
      zoneMap.set(z, {
        zone: z,
        cameraCount: 0,
        vehicleCount: 0,
        uniquePlateCount: 0,
        cameras: [],
      });
    }
    const entry = zoneMap.get(z);
    entry.cameraCount++;
    entry.vehicleCount += cd.vehicleCount;
    entry.uniquePlateCount += cd.uniquePlateCount;
    entry.cameras.push(cd.cameraId);
  });

  return Array.from(zoneMap.values()).map((z) => ({
    ...z,
    densityLevel: classifyDensityLevel(z.vehicleCount, options.config?.densityThresholds),
  }));
}

/**
 * Main handler function for GET /api/analytics/density
 */
function getDensityAnalytics(detections, query = {}) {
  // Validate time range
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  // Validate interval if provided
  let interval = '15m';
  if (query.interval) {
    const intRes = parseInterval(query.interval);
    if (!intRes.valid) {
      return { success: false, status: 400, message: intRes.message };
    }
    interval = intRes.interval;
  }

  const cfg = loadConfig();
  const options = {
    timeRange: timeRes,
    interval,
    cameraId: query.cameraId || null,
    config: cfg,
  };

  const cameraResults = getDensityByCamera(detections, options);
  const zoneResults = getDensityByZone(detections, options);

  const timeFiltered = filterDetectionsByTime(detections, timeRes.fromDate, timeRes.toDate);
  const uniquePlatesSet = new Set(
    timeFiltered
      .map((d) => (d.plate || '').toString().toUpperCase().replace(/[^A-Z0-9]/g, ''))
      .filter(Boolean)
  );

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    interval,
    totalDetections: timeFiltered.length,
    uniqueVehicles: uniquePlatesSet.size,
    cameras: cameraResults,
    zones: zoneResults,
  };
}

module.exports = {
  classifyDensityLevel,
  getVehicleCountByCamera,
  getDensityByCamera,
  getDensityByZone,
  getDensityAnalytics,
};
