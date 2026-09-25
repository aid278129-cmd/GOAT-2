'use strict';

/**
 * Congestion Analytics Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const { loadConfig, parseTimeRange } = require('./analyticsUtils');
const { calculateRouteDensity } = require('./routeAnalytics');

/**
 * Priority map for sorting congestion levels.
 */
const CONGESTION_PRIORITY = {
  SEVERE: 4,
  HIGH: 3,
  MODERATE: 2,
  NORMAL: 1,
};

/**
 * Calculates historical baseline metrics for each route across all available detections.
 */
function calculateHistoricalBaselines(allDetections, minEvaluationCount = 5) {
  const allRoutes = calculateRouteDensity(allDetections);
  const baselines = new Map();

  allRoutes.forEach((r) => {
    if (r.vehicleCount >= minEvaluationCount) {
      baselines.set(r.route, {
        baselineVehicleCount: r.vehicleCount,
        baselineAverageSpeedKmh: r.averageSpeedKmh,
        baselineSource: 'HISTORICAL',
      });
    }
  });

  return baselines;
}

/**
 * Evaluates congestion for routes in the current time window compared to baseline.
 *
 * @param {Array<object>} currentDetections
 * @param {Array<object>} allDetections
 * @param {object} [options]
 * @returns {Array<object>}
 */
function calculateCongestion(currentDetections, allDetections = null, options = {}) {
  const cfg = options.config || loadConfig();
  const congConfig = cfg.congestion || {};

  const densityMultiplierHigh = Number(congConfig.densityMultiplierHigh) || 1.5;
  const speedRatioHigh = Number(congConfig.speedRatioHigh) || 0.6;
  const minVehicles = Number(congConfig.minimumVehiclesForEvaluation) || 5;
  const defaultBaseSpeed = Number(congConfig.defaultBaselineSpeedKmh) || 40.0;
  const defaultBaseVolume = Number(congConfig.defaultBaselineVolumePerHour) || 250;

  // Compute baseline if allDetections provided, otherwise use current or config
  const baselines = allDetections && allDetections.length > 0
    ? calculateHistoricalBaselines(allDetections, minVehicles)
    : new Map();

  const currentRoutes = calculateRouteDensity(currentDetections, options);

  const results = currentRoutes.map((r) => {
    let baselineVehicleCount = defaultBaseVolume;
    let baselineAverageSpeedKmh = defaultBaseSpeed;
    let baselineSource = 'CONFIGURATION';

    if (baselines.has(r.route)) {
      const base = baselines.get(r.route);
      baselineVehicleCount = base.baselineVehicleCount;
      baselineAverageSpeedKmh = base.baselineAverageSpeedKmh;
      baselineSource = 'HISTORICAL';
    }

    const currentCount = r.vehicleCount;
    const currentSpeed = r.averageSpeedKmh;

    const countRatio = baselineVehicleCount > 0 ? currentCount / baselineVehicleCount : 1.0;
    const speedRatio = baselineAverageSpeedKmh > 0 ? currentSpeed / baselineAverageSpeedKmh : 1.0;

    let level = 'NORMAL';
    let reason = 'Traffic flow is within normal parameters';

    if (countRatio >= 2.0 && speedRatio <= 0.4) {
      level = 'SEVERE';
      reason = 'Extreme traffic volume surge with severe speed degradation';
    } else if (countRatio >= densityMultiplierHigh && speedRatio <= speedRatioHigh) {
      level = 'HIGH';
      reason = 'Significant volume increase combined with substantial speed drop';
    } else if (countRatio >= 1.2 || speedRatio <= 0.75) {
      level = 'MODERATE';
      reason = 'Moderate volume increase or noticeable speed reduction observed';
    }

    return {
      route: r.route,
      level,
      reason,
      fromCameraId: r.fromCameraId,
      toCameraId: r.toCameraId,
      fromCameraName: r.fromCameraName,
      toCameraName: r.toCameraName,
      fromLocation: r.fromLocation,
      toLocation: r.toLocation,
      fromLatitude: r.fromLatitude || null,
      fromLongitude: r.fromLongitude || null,
      toLatitude: r.toLatitude || null,
      toLongitude: r.toLongitude || null,
      fromCoordinates: r.fromCoordinates || null,
      toCoordinates: r.toCoordinates || null,
      currentVehicleCount: currentCount,
      baselineVehicleCount,
      currentAverageSpeedKmh: currentSpeed,
      baselineAverageSpeedKmh,
      volumeRatio: Number(countRatio.toFixed(2)),
      speedRatio: Number(speedRatio.toFixed(2)),
      baselineSource,
    };
  });

  // Sort by congestion severity descending (SEVERE -> HIGH -> MODERATE -> NORMAL), then volume
  results.sort((a, b) => {
    const diff = (CONGESTION_PRIORITY[b.level] || 1) - (CONGESTION_PRIORITY[a.level] || 1);
    if (diff !== 0) return diff;
    return b.currentVehicleCount - a.currentVehicleCount;
  });

  return results;
}

/**
 * Main handler function for GET /api/analytics/congestion
 */
function getCongestionAnalytics(detections, query = {}) {
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  const routes = calculateCongestion(detections, detections, {
    timeRange: timeRes,
  });

  const congestedCount = routes.filter((r) => r.level === 'HIGH' || r.level === 'SEVERE').length;

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    congestedRouteCount: congestedCount,
    totalEvaluatedRoutes: routes.length,
    routes,
  };
}

module.exports = {
  calculateHistoricalBaselines,
  calculateCongestion,
  getCongestionAnalytics,
};
