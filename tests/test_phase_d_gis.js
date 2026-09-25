/**
 * test_phase_d_gis.js — Automated Test Suite for Phase D GIS Traffic Visualization
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

'use strict';

const assert = require('assert');
const {
  normalizeHeatmapData,
  normalizeRouteWidth,
  mapCongestionStyle,
  mapDensityStyle,
  resolveRouteEndpoints,
  calculateTimeRangeQuery,
} = require('../public/js/trafficMap');

let totalTests = 0;
let passedTests = 0;

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

console.log('======================================================================');
console.log('🧪 RUNNING PHASE D GIS TRAFFIC VISUALIZATION TEST SUITE');
console.log('======================================================================');

// ─────────────────────────────────────────────────────────────────────────────
// 1. Heatmap Data Normalization Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 1. Heatmap Normalization Tests ---');

runTest('normalizeHeatmapData extracts valid coordinates and normalizes intensity against max', () => {
  const sampleCameras = [
    { cameraId: 'CAM_01', latitude: 13.0827, longitude: 80.2707, vehicleCount: 100 },
    { cameraId: 'CAM_02', latitude: 13.0731, longitude: 80.2609, vehicleCount: 50 },
    { cameraId: 'CAM_03', latitude: 13.0878, longitude: 80.2785, vehicleCount: 200 }, // Max
  ];

  const points = normalizeHeatmapData(sampleCameras);
  assert.strictEqual(points.length, 3);

  // Highest volume camera should have intensity 1.0
  const maxPt = points.find((p) => p[0] === 13.0878 && p[1] === 80.2785);
  assert.strictEqual(maxPt[2], 1.0, 'Highest volume node should have intensity 1.0');

  // Half volume camera (100 / 200) should have intensity 0.5
  const halfPt = points.find((p) => p[0] === 13.0827 && p[1] === 80.2707);
  assert.strictEqual(halfPt[2], 0.5, '100/200 count should normalize to 0.5');

  // Quarter volume (50 / 200 = 0.25)
  const quartPt = points.find((p) => p[0] === 13.0731 && p[1] === 80.2609);
  assert.strictEqual(quartPt[2], 0.25, '50/200 count should normalize to 0.25');
});

runTest('normalizeHeatmapData handles zero maximum count safely without division by zero', () => {
  const zeroCameras = [
    { cameraId: 'CAM_01', latitude: 13.0827, longitude: 80.2707, vehicleCount: 0 },
    { cameraId: 'CAM_02', latitude: 13.0731, longitude: 80.2609, vehicleCount: 0 },
  ];

  const points = normalizeHeatmapData(zeroCameras);
  assert.strictEqual(points.length, 2);
  assert.strictEqual(points[0][2], 0, 'Zero max count should produce 0 intensity');
  assert.strictEqual(points[1][2], 0, 'Zero max count should produce 0 intensity');
});

runTest('normalizeHeatmapData safely skips missing, null, and out-of-range coordinates', () => {
  const dirtyCameras = [
    { cameraId: 'CAM_01', latitude: 13.0827, longitude: 80.2707, vehicleCount: 50 },
    { cameraId: 'CAM_BAD1', latitude: null, longitude: 80.2609, vehicleCount: 90 },
    { cameraId: 'CAM_BAD2', latitude: 0, longitude: 0, vehicleCount: 40 },
    { cameraId: 'CAM_BAD3', latitude: 95.0, longitude: 80.0, vehicleCount: 30 }, // lat > 90
    { cameraId: 'CAM_02', lat: 13.0731, lng: 80.2609, vehicleCount: 80 }, // alternative lat/lng keys
  ];

  const points = normalizeHeatmapData(dirtyCameras);
  assert.strictEqual(points.length, 2, 'Should only contain the 2 valid cameras');
  assert.strictEqual(points[0][0], 13.0827);
  assert.strictEqual(points[1][0], 13.0731);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Route Density Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. Route Density & Width Normalization Tests ---');

runTest('normalizeRouteWidth scales line thickness smoothly between min and max pixels', () => {
  const maxVolume = 1000;

  // Max volume -> maxWidth (12px)
  assert.strictEqual(normalizeRouteWidth(1000, maxVolume, 2, 12), 12);

  // 0 volume -> minWidth (2px)
  assert.strictEqual(normalizeRouteWidth(0, maxVolume, 2, 12), 2);

  // Half volume -> midpoint (7px)
  assert.strictEqual(normalizeRouteWidth(500, maxVolume, 2, 12), 7);

  // Clamps properly even if count exceeds max
  assert.strictEqual(normalizeRouteWidth(1500, maxVolume, 2, 12), 12);
});

runTest('resolveRouteEndpoints extracts direct coordinates from route metadata', () => {
  const route = {
    route: 'CAM_01->CAM_02',
    fromCameraId: 'CAM_01',
    toCameraId: 'CAM_02',
    fromCoordinates: [13.0827, 80.2707],
    toCoordinates: [13.0731, 80.2609],
  };

  const endpoints = resolveRouteEndpoints(route);
  assert(endpoints !== null);
  assert.deepStrictEqual(endpoints.fromCoord, [13.0827, 80.2707]);
  assert.deepStrictEqual(endpoints.toCoord, [13.0731, 80.2609]);
});

runTest('resolveRouteEndpoints resolves coordinates via camera lookup table with ID normalization', () => {
  const cameraLookup = [
    { standardId: 'CAM_01', cameraId: 1, latitude: 13.0827, longitude: 80.2707 },
    { standardId: 'CAM_02', cameraId: 2, latitude: 13.0731, longitude: 80.2609 },
  ];

  // Route using numeric camera IDs
  const routeWithNumeric = {
    route: '1->2',
    fromCameraId: 1,
    toCameraId: 2,
  };

  const endpoints = resolveRouteEndpoints(routeWithNumeric, cameraLookup);
  assert(endpoints !== null);
  assert.deepStrictEqual(endpoints.fromCoord, [13.0827, 80.2707]);
  assert.deepStrictEqual(endpoints.toCoord, [13.0731, 80.2609]);
});

runTest('resolveRouteEndpoints returns null gracefully when camera coordinates are missing', () => {
  const missingRoute = {
    route: 'CAM_01->CAM_UNKNOWN',
    fromCameraId: 'CAM_01',
    toCameraId: 'CAM_UNKNOWN',
  };

  const endpoints = resolveRouteEndpoints(missingRoute, []);
  assert.strictEqual(endpoints, null, 'Should return null when endpoints cannot be resolved');
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Congestion & Density Styling Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Congestion & Density Mapping Tests ---');

runTest('mapCongestionStyle produces correct color, weight, and badge hierarchy', () => {
  const severe = mapCongestionStyle('SEVERE');
  assert.strictEqual(severe.color, '#ff3b30');
  assert(severe.weight >= 6, 'Severe weight should be prominent');
  assert(severe.zIndex > 900, 'Severe zIndex should be top-most');

  const high = mapCongestionStyle('HIGH');
  assert.strictEqual(high.color, '#ff6b00');

  const mod = mapCongestionStyle('MODERATE');
  assert.strictEqual(mod.color, '#ffb400');

  const norm = mapCongestionStyle('NORMAL');
  assert.strictEqual(norm.color, '#00ff41');

  // Case insensitive & default fallback
  assert.strictEqual(mapCongestionStyle('severe').color, '#ff3b30');
  assert.strictEqual(mapCongestionStyle('unknown').color, '#00ff41');
});

runTest('mapDensityStyle maps LOW, MEDIUM, HIGH, SEVERE to accessible color schemes', () => {
  assert.strictEqual(mapDensityStyle('LOW').color, '#00ff41');
  assert.strictEqual(mapDensityStyle('MEDIUM').color, '#ffb400');
  assert.strictEqual(mapDensityStyle('HIGH').color, '#ff6b00');
  assert.strictEqual(mapDensityStyle('SEVERE').color, '#ff3b30');
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Time Range Query Helper Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 4. Time Range Query Helper Tests ---');

const FIXED_NOW = new Date('2026-09-25T12:00:00.000Z').getTime();

runTest('calculateTimeRangeQuery computes exact ISO bounds for 15m, 30m, 1h, and today', () => {
  const res15 = calculateTimeRangeQuery('15m', null, null, FIXED_NOW);
  assert.strictEqual(res15.valid, true);
  assert.strictEqual(res15.from, new Date(FIXED_NOW - 15 * 60000).toISOString());
  assert.strictEqual(res15.to, new Date(FIXED_NOW).toISOString());

  const res30 = calculateTimeRangeQuery('30m', null, null, FIXED_NOW);
  assert.strictEqual(res30.from, new Date(FIXED_NOW - 30 * 60000).toISOString());

  const res1h = calculateTimeRangeQuery('1h', null, null, FIXED_NOW);
  assert.strictEqual(res1h.from, new Date(FIXED_NOW - 60 * 60000).toISOString());

  const resToday = calculateTimeRangeQuery('today', null, null, FIXED_NOW);
  assert.strictEqual(resToday.valid, true);
  assert(typeof resToday.from === 'string', 'Today from must be a string');
  assert(new Date(resToday.from).getTime() <= FIXED_NOW, 'Today from must be <= current time');
});

runTest('calculateTimeRangeQuery validates custom date ranges and rejects inverted or malformed ranges', () => {
  const validCustom = calculateTimeRangeQuery(
    'custom',
    '2026-09-25T10:00:00Z',
    '2026-09-25T11:00:00Z',
    FIXED_NOW
  );
  assert.strictEqual(validCustom.valid, true);
  assert.strictEqual(validCustom.from, '2026-09-25T10:00:00.000Z');
  assert.strictEqual(validCustom.to, '2026-09-25T11:00:00.000Z');

  // Inverted range (from > to)
  const inverted = calculateTimeRangeQuery(
    'custom',
    '2026-09-25T11:00:00Z',
    '2026-09-25T10:00:00Z',
    FIXED_NOW
  );
  assert.strictEqual(inverted.valid, false);
  assert(inverted.error.includes('earlier'));

  // Malformed date
  const malformed = calculateTimeRangeQuery('custom', 'not-a-date', '2026-09-25T10:00:00Z', FIXED_NOW);
  assert.strictEqual(malformed.valid, false);
  assert(malformed.error.includes('Malformed'));

  // Missing parameter
  const missing = calculateTimeRangeQuery('custom', null, '2026-09-25T10:00:00Z', FIXED_NOW);
  assert.strictEqual(missing.valid, false);
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Empty State & Robustness Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 5. Empty States & Robustness Tests ---');

runTest('Handles empty arrays for heatmap and routes without throwing exceptions', () => {
  assert.deepStrictEqual(normalizeHeatmapData([]), []);
  assert.deepStrictEqual(normalizeHeatmapData(null), []);
  assert.strictEqual(normalizeRouteWidth(0, 0), 2);
  assert.strictEqual(resolveRouteEndpoints(null), null);
});

// ─────────────────────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n======================================================================');
console.log(`🏁 TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
console.log('======================================================================');

if (passedTests !== totalTests) {
  process.exit(1);
}
