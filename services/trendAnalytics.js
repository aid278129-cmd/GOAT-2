'use strict';

/**
 * Traffic Trend Analytics Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const {
  loadConfig,
  formatStandardCameraId,
  parseTimeRange,
  parseInterval,
  filterDetectionsByTime,
  deduplicateDetections,
} = require('./analyticsUtils');
const { calculateSpeedAnalytics } = require('./speedAnalytics');
const { calculateCongestion } = require('./congestionAnalytics');

/**
 * Groups detections into chronological time buckets ('5m', '15m', '30m', '1h')
 * and calculates time-series metrics.
 *
 * @param {Array<object>} detections
 * @param {object} [options]
 * @returns {Array<object>}
 */
function calculateTrafficTrends(detections, options = {}) {
  const timeFiltered = options.timeRange && (options.timeRange.fromDate || options.timeRange.toDate)
    ? filterDetectionsByTime(detections, options.timeRange.fromDate, options.timeRange.toDate)
    : detections;

  const intervalInfo = options.intervalInfo || parseInterval(options.interval || '15m');
  const durationMs = intervalInfo.durationMs;

  if (!Array.isArray(timeFiltered) || timeFiltered.length === 0) {
    return [];
  }

  // Find min and max timestamps
  let minTime = Infinity;
  let maxTime = -Infinity;

  timeFiltered.forEach((d) => {
    const t = new Date(d.timestamp).getTime();
    if (!isNaN(t)) {
      if (t < minTime) minTime = t;
      if (t > maxTime) maxTime = t;
    }
  });

  if (!isFinite(minTime) || !isFinite(maxTime)) {
    return [];
  }

  // If explicit from/to provided in options, align bucket boundaries
  const startMs = options.timeRange?.fromDate
    ? Math.floor(options.timeRange.fromDate.getTime() / durationMs) * durationMs
    : Math.floor(minTime / durationMs) * durationMs;

  const endMs = options.timeRange?.toDate
    ? Math.ceil(options.timeRange.toDate.getTime() / durationMs) * durationMs
    : Math.ceil(maxTime / durationMs) * durationMs;

  const cfg = options.config || loadConfig();
  const buckets = [];

  for (let bStart = startMs; bStart <= endMs; bStart += durationMs) {
    const bEnd = bStart + durationMs;
    const bucketDets = timeFiltered.filter((d) => {
      const t = new Date(d.timestamp).getTime();
      return t >= bStart && t < bEnd;
    });

    const deduped = deduplicateDetections(bucketDets, cfg.duplicateDetectionWindowSeconds);

    const uniquePlates = new Set(
      deduped
        .map((d) => (d.plate || '').toString().toUpperCase().replace(/[^A-Z0-9]/g, ''))
        .filter(Boolean)
    );

    const activeCameras = new Set(
      deduped.map((d) => formatStandardCameraId(d.cameraId))
    );

    let averageSpeedKmh = null;
    let congestionCount = 0;

    if (bucketDets.length >= 2) {
      const speedRes = calculateSpeedAnalytics(bucketDets);
      averageSpeedKmh = speedRes.overallEstimatedAverageSpeedKmh;

      const congRes = calculateCongestion(bucketDets, detections);
      congestionCount = congRes.filter((c) => c.level === 'HIGH' || c.level === 'SEVERE').length;
    }

    buckets.push({
      timestamp: new Date(bStart).toISOString(),
      timeFormatted: new Date(bStart).toTimeString().slice(0, 5),
      interval: intervalInfo.interval,
      vehicleCount: deduped.length,
      rawDetectionCount: bucketDets.length,
      uniqueVehicles: uniquePlates.size,
      averageSpeedKmh,
      activeCameras: activeCameras.size,
      congestionCount,
    });
  }

  return buckets;
}

/**
 * Main handler function for GET /api/analytics/trends
 */
function getTrendAnalytics(detections, query = {}) {
  const timeRes = parseTimeRange(query.from, query.to);
  if (!timeRes.valid) {
    return { success: false, status: 400, message: timeRes.message };
  }

  const intRes = parseInterval(query.interval || '15m');
  if (!intRes.valid) {
    return { success: false, status: 400, message: intRes.message };
  }

  const trends = calculateTrafficTrends(detections, {
    timeRange: timeRes,
    intervalInfo: intRes,
    interval: intRes.interval,
  });

  return {
    success: true,
    from: timeRes.fromIso,
    to: timeRes.toIso,
    interval: intRes.interval,
    bucketCount: trends.length,
    trends,
  };
}

module.exports = {
  calculateTrafficTrends,
  getTrendAnalytics,
};
