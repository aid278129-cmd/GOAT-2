'use strict';

/**
 * Route Density Analytics Service
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
const { getRoadRouteSync } = require('./roadRoutingService');

/**
 * Extracts and aggregates all valid camera-to-camera route segments across vehicle trajectories.
 * Only segments with validForAnalytics === true are included.
 *
 * @param {Array<object>} detections
 * @param {object} [options]
 * @returns {Array<object>}
 */
function calculateRouteDensity(detections, options = {}) {
  const timeFiltered = options.timeRange && (options.timeRange.fromDate || options.timeRange.toDate)
    ? filterDetectionsByTime(detections, options.timeRange.fromDate, options.timeRange.toDate)
    : detections;

  if (!Array.isArray(timeFiltered) || timeFiltered.length < 2) {
    return [];
  }

  const cameras = loadCameras();

  // Group detections by normalized plate
  const plateMap = new Map();
  timeFiltered.forEach((d) => {
    const plate = trajectoryService.normalizePlate(d.plate);
    if (!plate) return;
    if (!plateMap.has(plate)) {
      plateMap.set(plate, []);
    }
    plateMap.get(plate).push(d);
  });

  const routeAggregates = new Map(); // key `${fromStandardId}->${toStandardId}`

  // Process trajectory for each plate
  plateMap.forEach((dets, plate) => {
    // Sort chronologically
    const sorted = [...dets].sort((a, b) => {
      const tA = new Date(a.timestamp).getTime();
      const tB = new Date(b.timestamp).getTime();
      return (isNaN(tA) ? 0 : tA) - (isNaN(tB) ? 0 : tB);
    });

    // Sessionize into discrete journeys to prevent merging separate trips
    const journeys = trajectoryService.segmentSightingsIntoJourneys(sorted, options.config);

    journeys.forEach((journey) => {
      const segments = journey.segments || [];

      segments.forEach((seg) => {
        // Respect data quality rules: strictly valid segments only
        if (!seg.validForAnalytics || seg.isSameCamera || seg.anomaly) {
          return;
        }

        if (seg.distanceKm === null || seg.travelTimeSeconds === null || seg.travelTimeSeconds <= 0) {
          return;
        }

        const fromCam = resolveCamera(seg.fromCameraId, cameras);
        const toCam = resolveCamera(seg.toCameraId, cameras);

        const fromId = fromCam.standardId;
        const toId = toCam.standardId;
        const routeKey = `${fromId}->${toId}`;

        // Prefer road network distance if available
        let segmentDistanceKm = seg.rawDistanceKm !== undefined ? seg.rawDistanceKm : seg.distanceKm;
        const roadRoute = getRoadRouteSync(fromId, toId);
        if (roadRoute && roadRoute.roadAligned && roadRoute.distanceKm > 0) {
          segmentDistanceKm = roadRoute.distanceKm;
        }

        if (!routeAggregates.has(routeKey)) {
          routeAggregates.set(routeKey, {
            route: routeKey,
            fromCameraId: fromId,
            toCameraId: toId,
            fromCameraNumericId: fromCam.cameraId,
            toCameraNumericId: toCam.cameraId,
            fromCameraName: fromCam.name,
            toCameraName: toCam.name,
            fromLocation: fromCam.roadName,
            toLocation: toCam.roadName,
            fromLatitude: fromCam.latitude || null,
            fromLongitude: fromCam.longitude || null,
            toLatitude: toCam.latitude || null,
            toLongitude: toCam.longitude || null,
            fromCoordinates: (fromCam.latitude && fromCam.longitude) ? [fromCam.latitude, fromCam.longitude] : null,
            toCoordinates: (toCam.latitude && toCam.longitude) ? [toCam.latitude, toCam.longitude] : null,
            vehicleCount: 0,
            uniquePlates: new Set(),
            totalDistanceKm: 0,
            totalTravelTimeSeconds: 0,
          });
        }

        const agg = routeAggregates.get(routeKey);
        agg.vehicleCount++;
        agg.uniquePlates.add(plate);
        agg.totalDistanceKm += segmentDistanceKm;
        agg.totalTravelTimeSeconds += seg.travelTimeSeconds;
      });
    });
  });

  const results = Array.from(routeAggregates.values()).map((agg) => {
    const hours = agg.totalTravelTimeSeconds / 3600;
    const avgSpeed = hours > 0 ? Number((agg.totalDistanceKm / hours).toFixed(1)) : 0;
    const avgTime = agg.vehicleCount > 0 ? Math.round(agg.totalTravelTimeSeconds / agg.vehicleCount) : 0;
    const avgDist = agg.vehicleCount > 0 ? Number((agg.totalDistanceKm / agg.vehicleCount).toFixed(2)) : 0;

    return {
      route: agg.route,
      fromCameraId: agg.fromCameraId,
      toCameraId: agg.toCameraId,
      fromCameraNumericId: agg.fromCameraNumericId,
      toCameraNumericId: agg.toCameraNumericId,
      fromCameraName: agg.fromCameraName,
      toCameraName: agg.toCameraName,
      fromLocation: agg.fromLocation,
      toLocation: agg.toLocation,
      fromLatitude: agg.fromLatitude,
      fromLongitude: agg.fromLongitude,
      toLatitude: agg.toLatitude,
      toLongitude: agg.toLongitude,
      fromCoordinates: agg.fromCoordinates,
      toCoordinates: agg.toCoordinates,
      vehicleCount: agg.vehicleCount,
      uniqueVehicles: agg.uniquePlates.size,
      averageDistanceKm: avgDist,
      averageTravelTimeSeconds: avgTime,
      averageTravelTimeMinutes: Number((avgTime / 60).toFixed(1)),
      averageSpeedKmh: avgSpeed,
    };
  });

  // Sort descending by vehicle volume
  results.sort((a, b) => b.vehicleCount - a.vehicleCount);

  if (options.limit && Number(options.limit) > 0) {
    return results.slice(0, Number(options.limit));
  }

  return results;
}

/**
 * Main handler function for GET /api/analytics/routes
 */
function getRouteDensityAnalytics(detections, query = {}) {
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  const limit = query.limit ? parseInt(query.limit, 10) : null;
  const routes = calculateRouteDensity(detections, {
    timeRange: timeRes,
    limit,
  });

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    routeCount: routes.length,
    routes,
  };
}

module.exports = {
  calculateRouteDensity,
  getRouteDensityAnalytics,
};
