'use strict';

/**
 * Average Speed Analytics Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const {
  formatStandardCameraId,
  parseTimeRange,
} = require('./analyticsUtils');
const { calculateRouteDensity } = require('./routeAnalytics');

/**
 * Calculates weighted estimated average speed across routes and city-wide network.
 * Formula: total valid distance / total valid travel time in hours.
 *
 * @param {Array<object>} detections
 * @param {object} [options]
 * @returns {object}
 */
function calculateSpeedAnalytics(detections, options = {}) {
  const routes = calculateRouteDensity(detections, options);

  let filteredRoutes = routes;
  if (options.fromCameraId) {
    const fId = formatStandardCameraId(options.fromCameraId);
    filteredRoutes = filteredRoutes.filter((r) => r.fromCameraId === fId);
  }
  if (options.toCameraId) {
    const tId = formatStandardCameraId(options.toCameraId);
    filteredRoutes = filteredRoutes.filter((r) => r.toCameraId === tId);
  }

  let totalDistanceKm = 0;
  let totalTravelTimeSeconds = 0;
  let totalJourneys = 0;

  const routeSpeeds = filteredRoutes.map((r) => {
    const routeTotalDist = (r.averageDistanceKm || 0) * r.vehicleCount;
    const routeTotalTime = (r.averageTravelTimeSeconds || 0) * r.vehicleCount;

    totalDistanceKm += routeTotalDist;
    totalTravelTimeSeconds += routeTotalTime;
    totalJourneys += r.vehicleCount;

    return {
      route: r.route,
      fromCameraId: r.fromCameraId,
      toCameraId: r.toCameraId,
      fromCameraName: r.fromCameraName,
      toCameraName: r.toCameraName,
      fromLocation: r.fromLocation,
      toLocation: r.toLocation,
      vehicleCount: r.vehicleCount,
      estimatedAverageSpeedKmh: r.averageSpeedKmh,
      averageSpeedKmh: r.averageSpeedKmh, // shorthand
      averageTravelTimeSeconds: r.averageTravelTimeSeconds,
      averageDistanceKm: r.averageDistanceKm,
    };
  });

  // Calculate weighted overall average speed
  let overallEstimatedAverageSpeedKmh = null;
  if (totalDistanceKm > 0 && totalTravelTimeSeconds > 0) {
    const totalHours = totalTravelTimeSeconds / 3600;
    overallEstimatedAverageSpeedKmh = Number((totalDistanceKm / totalHours).toFixed(1));
  }

  return {
    routeCount: routeSpeeds.length,
    overallEstimatedAverageSpeedKmh,
    totalValidJourneys: totalJourneys,
    routes: routeSpeeds,
  };
}

/**
 * Main handler function for GET /api/analytics/speed
 */
function getSpeedAnalytics(detections, query = {}) {
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  const result = calculateSpeedAnalytics(detections, {
    timeRange: timeRes,
    fromCameraId: query.fromCameraId || null,
    toCameraId: query.toCameraId || null,
  });

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    ...result,
  };
}

module.exports = {
  calculateSpeedAnalytics,
  getSpeedAnalytics,
};
