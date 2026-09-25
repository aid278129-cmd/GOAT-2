'use strict';

/**
 * Phase C — Macro Traffic Flow & Movement Analytics Comprehensive Test Suite
 * Validates Traffic Density, Route Density, Speed, OD Matrix, Congestion, and Trends.
 */

const assert = require('assert');
const analyticsUtils = require('../services/analyticsUtils');
const trafficAnalytics = require('../services/trafficAnalytics');
const routeAnalytics = require('../services/routeAnalytics');
const speedAnalytics = require('../services/speedAnalytics');
const odAnalytics = require('../services/odAnalytics');
const congestionAnalytics = require('../services/congestionAnalytics');
const trendAnalytics = require('../services/trendAnalytics');
const analyticsService = require('../services/analyticsService');

console.log('======================================================================');
console.log('🧪 RUNNING PHASE C TRAFFIC FLOW & MOVEMENT ANALYTICS TEST SUITE');
console.log('======================================================================\n');

let passedTests = 0;
let totalTests = 0;

function runTest(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  ✅ PASS: ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${name}`);
    console.error(`     Error: ${err.message}`);
    if (err.stack) {
      console.error(`     ${err.stack.split('\n').slice(1, 3).join('\n     ')}`);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic Test Dataset (4 cameras, 10+ plates, multi-route journeys)
// ─────────────────────────────────────────────────────────────────────────────
// Cameras:
// 1 / CAM_01: Junction A (13.0827, 80.2707)
// 2 / CAM_02: Junction B (13.0731, 80.2609)  [dist from CAM_01 ~ 1.51 km]
// 3 / CAM_03: Junction C (13.0878, 80.2785)  [dist from CAM_02 ~ 2.47 km]
// 4 / CAM_04: Highway Entry (13.0569, 80.2425)

const BASE_TIME = new Date('2026-09-25T10:00:00Z').getTime();

const testDetections = [
  // Vehicle 1: Standard journey CAM_01 (10:00) -> CAM_02 (10:05 -> 5 min, ~18.1 km/h) -> CAM_03 (10:12 -> 7 min)
  { id: 'v1-1', plate: 'TN01AA1111', cameraId: 'CAM_01', timestamp: new Date(BASE_TIME).toISOString(), latitude: 13.0827, longitude: 80.2707 },
  { id: 'v1-2', plate: 'TN01AA1111', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 5 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },
  { id: 'v1-3', plate: 'TN01AA1111', cameraId: 'CAM_03', timestamp: new Date(BASE_TIME + 12 * 60000).toISOString(), latitude: 13.0878, longitude: 80.2785 },

  // Vehicle 2: Duplicate burst at CAM_01 (10:00:00, 10:00:02, 10:00:05), then moves to CAM_02 (10:08)
  { id: 'v2-1', plate: 'TN02BB2222', cameraId: 1, timestamp: new Date(BASE_TIME).toISOString(), lat: 13.0827, lng: 80.2707 },
  { id: 'v2-2', plate: 'TN02BB2222', cameraId: 1, timestamp: new Date(BASE_TIME + 2000).toISOString(), lat: 13.0827, lng: 80.2707 }, // Duplicate
  { id: 'v2-3', plate: 'TN02BB2222', cameraId: 1, timestamp: new Date(BASE_TIME + 5000).toISOString(), lat: 13.0827, lng: 80.2707 }, // Duplicate
  { id: 'v2-4', plate: 'TN02BB2222', cameraId: 2, timestamp: new Date(BASE_TIME + 8 * 60000).toISOString(), lat: 13.0731, lng: 80.2609 },

  // Vehicle 3: Journey CAM_01 (10:02) -> CAM_02 (10:07)
  { id: 'v3-1', plate: 'TN03CC3333', cameraId: 'CAM_01', timestamp: new Date(BASE_TIME + 2 * 60000).toISOString(), latitude: 13.0827, longitude: 80.2707 },
  { id: 'v3-2', plate: 'TN03CC3333', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 7 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },

  // Vehicle 4: Journey CAM_01 (10:04) -> CAM_02 (10:10)
  { id: 'v4-1', plate: 'KA04DD4444', cameraId: 'CAM_01', timestamp: new Date(BASE_TIME + 4 * 60000).toISOString(), latitude: 13.0827, longitude: 80.2707 },
  { id: 'v4-2', plate: 'KA04DD4444', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 10 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },

  // Vehicle 5: Congested Slow Journey CAM_02 (10:10) -> CAM_03 (10:35 -> 25 mins for ~2.5km -> ~6 km/h)
  { id: 'v5-1', plate: 'MH05EE5555', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 10 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },
  { id: 'v5-2', plate: 'MH05EE5555', cameraId: 'CAM_03', timestamp: new Date(BASE_TIME + 35 * 60000).toISOString(), latitude: 13.0878, longitude: 80.2785 },

  // Vehicle 6: Impossible Speed Journey (CAM_01 -> CAM_04 within 2 seconds! > 1000 km/h)
  { id: 'v6-1', plate: 'FAST9999', cameraId: 'CAM_01', timestamp: new Date(BASE_TIME + 15 * 60000).toISOString(), latitude: 13.0827, longitude: 80.2707 },
  { id: 'v6-2', plate: 'FAST9999', cameraId: 'CAM_04', timestamp: new Date(BASE_TIME + 15 * 60000 + 2000).toISOString(), latitude: 13.0569, longitude: 80.2425 },

  // Vehicle 7: Journey CAM_03 -> CAM_04
  { id: 'v7-1', plate: 'DL07GG7777', cameraId: 'CAM_03', timestamp: new Date(BASE_TIME + 16 * 60000).toISOString(), latitude: 13.0878, longitude: 80.2785 },
  { id: 'v7-2', plate: 'DL07GG7777', cameraId: 'CAM_04', timestamp: new Date(BASE_TIME + 28 * 60000).toISOString(), latitude: 13.0569, longitude: 80.2425 },

  // Vehicle 8: Journey CAM_02 -> CAM_04
  { id: 'v8-1', plate: 'KL08HH8888', cameraId: 2, timestamp: new Date(BASE_TIME + 20 * 60000).toISOString(), lat: 13.0731, lng: 80.2609 },
  { id: 'v8-2', plate: 'KL08HH8888', cameraId: 4, timestamp: new Date(BASE_TIME + 32 * 60000).toISOString(), lat: 13.0569, lng: 80.2425 },

  // Vehicle 9: Single sighting only (CAM_04 at 10:25)
  { id: 'v9-1', plate: 'HR09JJ9999', cameraId: 'CAM_04', timestamp: new Date(BASE_TIME + 25 * 60000).toISOString(), latitude: 13.0569, longitude: 80.2425 },

  // Vehicle 10: Same camera stationary (CAM_02 at 10:30 and 10:32)
  { id: 'v10-1', plate: 'AP10KK1010', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 30 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },
  { id: 'v10-2', plate: 'AP10KK1010', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 32 * 60000).toISOString(), latitude: 13.0731, longitude: 80.2609 },

  // Vehicle 11: Malformed timestamp & missing coordinate records to test safety
  { id: 'v11-bad', plate: 'BAD0000', cameraId: 'CAM_01', timestamp: 'invalid-time', latitude: 13.0827, longitude: 80.2707 },
  { id: 'v12-bad', plate: 'NOCOORD', cameraId: 'CAM_02', timestamp: new Date(BASE_TIME + 35 * 60000).toISOString(), latitude: null, longitude: null },
];

// ─────────────────────────────────────────────────────────────────────────────
// 1. Traffic Density Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('--- 1. Traffic Density Tests ---');

runTest('deduplicateDetections filters multi-frame burst detections of same plate', () => {
  const deduped = analyticsUtils.deduplicateDetections(testDetections, 10);
  const v2Count = deduped.filter((d) => d.plate === 'TN02BB2222' && (d.cameraId === 1 || d.cameraId === 'CAM_01')).length;
  assert.strictEqual(v2Count, 1, `Expected 1 deduplicated sighting for TN02BB2222 at CAM_01, got ${v2Count}`);
});

runTest('getVehicleCountByCamera accurately tallies vehicle volume and unique plates per node', () => {
  const { countMap, platesMap } = trafficAnalytics.getVehicleCountByCamera(testDetections);
  assert(countMap.get('CAM_01') >= 5, `Expected >= 5 vehicles at CAM_01, got ${countMap.get('CAM_01')}`);
  assert(platesMap.get('CAM_01').has('TN01AA1111'), 'Plates map should contain TN01AA1111');
  assert(platesMap.get('CAM_01').has('TN02BB2222'), 'Plates map should contain TN02BB2222');
});

runTest('classifyDensityLevel correctly assigns LOW, MEDIUM, HIGH, SEVERE tiers', () => {
  assert.strictEqual(trafficAnalytics.classifyDensityLevel(50), 'LOW');
  assert.strictEqual(trafficAnalytics.classifyDensityLevel(150), 'MEDIUM');
  assert.strictEqual(trafficAnalytics.classifyDensityLevel(350), 'HIGH');
  assert.strictEqual(trafficAnalytics.classifyDensityLevel(650), 'SEVERE');
});

runTest('getDensityAnalytics supports time window filtering and cameraId filtering', () => {
  const from = new Date(BASE_TIME).toISOString();
  const to = new Date(BASE_TIME + 10 * 60000).toISOString();
  const res = trafficAnalytics.getDensityAnalytics(testDetections, { from, to, cameraId: 'CAM_01' });

  assert.strictEqual(res.success, true);
  assert.strictEqual(res.cameras.length, 1);
  assert.strictEqual(res.cameras[0].cameraId, 'CAM_01');
  assert(res.cameras[0].vehicleCount > 0);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Route Density Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. Route Density Tests ---');

runTest('calculateRouteDensity correctly identifies CAM_01->CAM_02 as busiest route', () => {
  const routes = routeAnalytics.calculateRouteDensity(testDetections);
  assert(routes.length >= 3, `Expected at least 3 routes, found ${routes.length}`);

  const topRoute = routes[0];
  assert.strictEqual(topRoute.route, 'CAM_01->CAM_02');
  assert.strictEqual(topRoute.fromCameraId, 'CAM_01');
  assert.strictEqual(topRoute.toCameraId, 'CAM_02');
  assert(topRoute.vehicleCount >= 3, `Expected >= 3 vehicles on CAM_01->CAM_02, found ${topRoute.vehicleCount}`);
});

runTest('calculateRouteDensity excludes impossible speed, same camera, and corrupted segments', () => {
  const routes = routeAnalytics.calculateRouteDensity(testDetections);
  // FAST9999 had impossible speed between CAM_01 and CAM_04
  const impossibleRoute = routes.find((r) => r.route === 'CAM_01->CAM_04');
  assert.strictEqual(impossibleRoute, undefined, 'Impossible travel route should be excluded from valid route density');

  // Same camera CAM_02->CAM_02 must not exist in route table
  const sameCamRoute = routes.find((r) => r.route === 'CAM_02->CAM_02');
  assert.strictEqual(sameCamRoute, undefined, 'Same-camera observation must be excluded');
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Speed Analytics Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Speed Analytics Tests ---');

runTest('calculateSpeedAnalytics produces weighted average speed across network', () => {
  const speedRes = speedAnalytics.calculateSpeedAnalytics(testDetections);
  assert(speedRes.overallEstimatedAverageSpeedKmh !== null);
  assert(speedRes.overallEstimatedAverageSpeedKmh > 5 && speedRes.overallEstimatedAverageSpeedKmh < 60,
    `Overall speed was ${speedRes.overallEstimatedAverageSpeedKmh} km/h`);
  assert(speedRes.totalValidJourneys >= 5);
});

runTest('calculateSpeedAnalytics filters by camera endpoints', () => {
  const speedRes = speedAnalytics.calculateSpeedAnalytics(testDetections, {
    fromCameraId: 'CAM_01',
    toCameraId: 'CAM_02',
  });
  assert.strictEqual(speedRes.routes.length, 1);
  assert.strictEqual(speedRes.routes[0].route, 'CAM_01->CAM_02');
  assert(speedRes.routes[0].averageSpeedKmh > 10 && speedRes.routes[0].averageSpeedKmh < 30);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Origin-Destination (OD) Matrix Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 4. Origin-Destination (OD) Matrix Tests ---');

runTest('calculateOriginDestination identifies first and last cameras of multi-point journeys', () => {
  const odRes = odAnalytics.calculateOriginDestination(testDetections);
  assert(odRes.totalJourneysEvaluated >= 5, `Expected >= 5 journeys, got ${odRes.totalJourneysEvaluated}`);

  // Vehicle 1 started at CAM_01 and ended at CAM_03
  const v1Pair = odRes.pairs.find((p) => p.originCameraId === 'CAM_01' && p.destinationCameraId === 'CAM_03');
  assert(v1Pair !== undefined, 'Should detect CAM_01->CAM_03 OD pair for TN01AA1111');
});

runTest('calculateOriginDestination builds square matrix with zero on diagonal by default', () => {
  const odRes = odAnalytics.calculateOriginDestination(testDetections);
  const matrix = odRes.matrix.grid;
  assert(Array.isArray(matrix), 'Matrix should be an array');
  assert.strictEqual(matrix.length, odRes.matrix.cameras.length);

  for (let i = 0; i < matrix.length; i++) {
    assert.strictEqual(matrix[i].length, odRes.matrix.cameras.length);
    assert.strictEqual(matrix[i][i], 0, `Diagonal entry at [${i}][${i}] must be 0`);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Congestion Detection Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 5. Congestion Detection Tests ---');

runTest('calculateCongestion identifies routes with abnormal speed drops or surges', () => {
  const cong = congestionAnalytics.calculateCongestion(testDetections);
  assert(Array.isArray(cong), 'Congestion output must be an array');
  assert(cong.length > 0, 'Should evaluate routes');

  // Verify each route has level and ratios
  cong.forEach((r) => {
    assert(['NORMAL', 'MODERATE', 'HIGH', 'SEVERE'].includes(r.level));
    assert(typeof r.currentVehicleCount === 'number');
    assert(typeof r.baselineVehicleCount === 'number');
    assert(typeof r.volumeRatio === 'number');
    assert(typeof r.speedRatio === 'number');
  });

  // Verify sorted by congestion severity
  const levels = cong.map((r) => r.level);
  for (let i = 1; i < levels.length; i++) {
    const priority = { SEVERE: 4, HIGH: 3, MODERATE: 2, NORMAL: 1 };
    assert(priority[levels[i - 1]] >= priority[levels[i]], 'Routes should be sorted by congestion priority descending');
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Traffic Trends Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 6. Traffic Trends Tests ---');

runTest('calculateTrafficTrends generates uniform time buckets (5m, 15m, 30m, 1h)', () => {
  ['5m', '15m', '30m', '1h'].forEach((interval) => {
    const trends = trendAnalytics.calculateTrafficTrends(testDetections, { interval });
    assert(trends.length > 0, `Expected buckets for interval ${interval}`);
    assert(trends[0].timestamp !== undefined);
    assert(typeof trends[0].vehicleCount === 'number');
    assert(typeof trends[0].uniqueVehicles === 'number');
    assert(typeof trends[0].activeCameras === 'number');
  });
});

runTest('getTrendAnalytics rejects unsupported interval with 400 status', () => {
  const res = trendAnalytics.getTrendAnalytics(testDetections, { interval: '45m' });
  assert.strictEqual(res.success, false);
  assert.strictEqual(res.status, 400);
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Executive Analytics Summary Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 7. Executive Analytics Summary Tests ---');

runTest('analyticsService.getSummary produces required dashboard card metrics', () => {
  const summary = analyticsService.getSummary(testDetections, [{ id: 'alert-1', plate: 'BOLO123' }], { 1: 'connected', 2: 'connected' });
  assert.strictEqual(summary.success, true);
  assert(summary.activeCameras >= 2, `Active cameras: ${summary.activeCameras}`);
  assert(typeof summary.vehiclesLastHour === 'number');
  assert(typeof summary.uniqueVehiclesLastHour === 'number');
  assert(typeof summary.congestedRoutes === 'number');
  assert.strictEqual(summary.watchlistAlerts, 1);
  assert(summary.invalidTrajectorySegments >= 1, 'Should record the impossible speed segment');
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. Edge Cases & Robustness
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 8. Edge Cases & Robustness ---');

runTest('Handles empty detections array gracefully without throwing', () => {
  const dRes = trafficAnalytics.getDensityAnalytics([]);
  assert.strictEqual(dRes.success, true);
  assert.strictEqual(dRes.totalDetections, 0);

  const rRes = routeAnalytics.getRouteDensityAnalytics([]);
  assert.strictEqual(rRes.success, true);
  assert.strictEqual(rRes.routes.length, 0);

  const sRes = speedAnalytics.getSpeedAnalytics([]);
  assert.strictEqual(sRes.success, true);
  assert.strictEqual(sRes.overallEstimatedAverageSpeedKmh, null);

  const odRes = odAnalytics.getOriginDestinationAnalytics([]);
  assert.strictEqual(odRes.success, true);
  assert.strictEqual(odRes.pairs.length, 0);

  const tRes = trendAnalytics.getTrendAnalytics([]);
  assert.strictEqual(tRes.success, true);
  assert.strictEqual(tRes.trends.length, 0);
});

runTest('Handles single detection without errors', () => {
  const single = [{ plate: 'SINGLE1', cameraId: 'CAM_01', timestamp: new Date().toISOString() }];
  const rRes = routeAnalytics.getRouteDensityAnalytics(single);
  assert.strictEqual(rRes.routes.length, 0);

  const odRes = odAnalytics.getOriginDestinationAnalytics(single);
  assert.strictEqual(odRes.pairs.length, 0);
});

runTest('Rejects malformed ISO timestamp queries with clean 400', () => {
  const res = trafficAnalytics.getDensityAnalytics(testDetections, { from: 'not-a-timestamp' });
  assert.strictEqual(res.success, false);
  assert.strictEqual(res.status, 400);

  const res2 = routeAnalytics.getRouteDensityAnalytics(testDetections, { from: '2026-10-01', to: '2026-09-01' });
  assert.strictEqual(res2.success, false);
  assert.strictEqual(res2.status, 400);
});

runTest('Normalizes numeric camera IDs and string camera IDs consistently', () => {
  const cam1 = analyticsUtils.resolveCamera(1);
  const cam2 = analyticsUtils.resolveCamera('CAM_01');
  const cam3 = analyticsUtils.resolveCamera('CAM-1');
  assert.strictEqual(cam1.standardId, 'CAM_01');
  assert.strictEqual(cam2.standardId, 'CAM_01');
  assert.strictEqual(cam3.standardId, 'CAM_01');
});

console.log('\n======================================================================');
console.log(`🏁 TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
console.log('======================================================================');

if (passedTests !== totalTests) {
  process.exit(1);
}
