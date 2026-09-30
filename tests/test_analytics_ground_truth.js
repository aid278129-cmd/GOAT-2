'use strict';

/**
 * Controlled Analytics Ground-Truth Test Suite
 * Mathematical verification against data/testing/analytics_ground_truth.json
 */

const assert = require('assert');
const path = require('path');
const fs = require('fs');

const {
  trafficAnalytics,
  routeAnalytics,
  speedAnalytics,
  odAnalytics,
  congestionAnalytics,
  trendAnalytics,
} = require('../services/analyticsService');
const { normalizeHeatmapData } = require('../public/js/trafficMap');

const GROUND_TRUTH_FILE = path.join(__dirname, '..', 'data', 'testing', 'analytics_ground_truth.json');
const detections = JSON.parse(fs.readFileSync(GROUND_TRUTH_FILE, 'utf8'));

console.log('======================================================================');
console.log('🧪 RUNNING CONTROLLED ANALYTICS GROUND-TRUTH TEST SUITE');
console.log('======================================================================');

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
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Traffic Density & Unique Vehicles
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 1. Traffic Density & Unique Vehicles ---');

const densityRes = trafficAnalytics.getDensityAnalytics(detections, {
  from: '2026-09-28T10:00:00.000Z',
  to: '2026-09-28T11:00:00.000Z',
  interval: '1h',
});

const cam1 = densityRes.cameras.find((c) => c.cameraId === 'CAM_01');
const cam2 = densityRes.cameras.find((c) => c.cameraId === 'CAM_02');
const cam3 = densityRes.cameras.find((c) => c.cameraId === 'CAM_03');

runTest('CAM_01: Exactly 12 raw detections, 10 deduplicated sightings, 10 unique vehicles', () => {
  assert.strictEqual(cam1.totalDetections, 12, `Expected 12 raw detections at CAM_01, got ${cam1.totalDetections}`);
  assert.strictEqual(cam1.vehicleCount, 10, `Expected 10 deduplicated vehicles at CAM_01, got ${cam1.vehicleCount}`);
  assert.strictEqual(cam1.uniquePlateCount, 10, `Expected 10 unique plates at CAM_01, got ${cam1.uniquePlateCount}`);
});

runTest('CAM_02: Exactly 21 raw detections, 20 deduplicated sightings, 20 unique vehicles', () => {
  assert.strictEqual(cam2.totalDetections, 21, `Expected 21 raw detections at CAM_02, got ${cam2.totalDetections}`);
  assert.strictEqual(cam2.vehicleCount, 20, `Expected 20 deduplicated vehicles at CAM_02, got ${cam2.vehicleCount}`);
  assert.strictEqual(cam2.uniquePlateCount, 20, `Expected 20 unique plates at CAM_02, got ${cam2.uniquePlateCount}`);
});

runTest('CAM_03: Exactly 5 raw detections, 5 deduplicated sightings, 5 unique vehicles', () => {
  assert.strictEqual(cam3.totalDetections, 5, `Expected 5 raw detections at CAM_03, got ${cam3.totalDetections}`);
  assert.strictEqual(cam3.vehicleCount, 5, `Expected 5 deduplicated vehicles at CAM_03, got ${cam3.vehicleCount}`);
  assert.strictEqual(cam3.uniquePlateCount, 5, `Expected 5 unique plates at CAM_03, got ${cam3.uniquePlateCount}`);
});

runTest('Busiest camera is correctly identified as CAM_02', () => {
  const sortedCams = [...densityRes.cameras].sort((a, b) => b.vehicleCount - a.vehicleCount);
  assert.strictEqual(sortedCams[0].cameraId, 'CAM_02');
  assert.strictEqual(sortedCams[0].vehicleCount, 20);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Vehicle Flow Rates
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. Vehicle Flow Rates ---');

runTest('CAM_01 flow rate: 0.17 veh/min, 0.83 veh/5min, 10.0 veh/h', () => {
  assert.strictEqual(cam1.flowRatePerMinute, 0.17);
  assert.strictEqual(cam1.flowRatePer5Min, 0.83);
  assert.strictEqual(cam1.flowRatePerHour, 10.0);
});

runTest('CAM_02 flow rate: 0.33 veh/min, 1.67 veh/5min, 20.0 veh/h', () => {
  assert.strictEqual(cam2.flowRatePerMinute, 0.33);
  assert.strictEqual(cam2.flowRatePer5Min, 1.67);
  assert.strictEqual(cam2.flowRatePerHour, 20.0);
});

runTest('CAM_03 flow rate: 0.08 veh/min, 0.42 veh/5min, 5.0 veh/h', () => {
  assert.strictEqual(cam3.flowRatePerMinute, 0.08);
  assert.strictEqual(cam3.flowRatePer5Min, 0.42);
  assert.strictEqual(cam3.flowRatePerHour, 5.0);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Route / Corridor Density & Transitions
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Route / Corridor Density & Transitions ---');

const routes = routeAnalytics.calculateRouteDensity(detections);

runTest('CAM_01 -> CAM_02 has exactly 8 valid transitions', () => {
  const r12 = routes.find((r) => r.route === 'CAM_01->CAM_02');
  assert(r12 !== undefined, 'Route CAM_01->CAM_02 should exist');
  assert.strictEqual(r12.vehicleCount, 8, `Expected 8 vehicles, got ${r12.vehicleCount}`);
  assert.strictEqual(r12.uniqueVehicles, 8);
});

runTest('CAM_02 -> CAM_03 has exactly 3 valid transitions', () => {
  const r23 = routes.find((r) => r.route === 'CAM_02->CAM_03');
  assert(r23 !== undefined, 'Route CAM_02->CAM_03 should exist');
  assert.strictEqual(r23.vehicleCount, 3, `Expected 3 vehicles, got ${r23.vehicleCount}`);
  assert.strictEqual(r23.uniqueVehicles, 3);
});

runTest('CAM_01 -> CAM_03 has exactly 2 valid transitions', () => {
  const r13 = routes.find((r) => r.route === 'CAM_01->CAM_03');
  assert(r13 !== undefined, 'Route CAM_01->CAM_03 should exist');
  assert.strictEqual(r13.vehicleCount, 2, `Expected 2 vehicles, got ${r13.vehicleCount}`);
  assert.strictEqual(r13.uniqueVehicles, 2);
});

runTest('Busiest route is CAM_01 -> CAM_02', () => {
  assert.strictEqual(routes[0].route, 'CAM_01->CAM_02');
  assert.strictEqual(routes[0].vehicleCount, 8);
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Origin-Destination (OD) Matrix
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 4. Origin-Destination (OD) Matrix ---');

const odRes = odAnalytics.calculateOriginDestination(detections);

runTest('Origin-Destination Matrix identifies True Origins and Final Destinations', () => {
  const od12 = odRes.pairs.find((p) => p.originCameraId === 'CAM_01' && p.destinationCameraId === 'CAM_02');
  const od13 = odRes.pairs.find((p) => p.originCameraId === 'CAM_01' && p.destinationCameraId === 'CAM_03');
  const od23 = odRes.pairs.find((p) => p.originCameraId === 'CAM_02' && p.destinationCameraId === 'CAM_03');

  // V01..V03 travel CAM_01 -> CAM_02 -> CAM_03. Origin is CAM_01, Destination is CAM_03!
  // V04..V08 travel CAM_01 -> CAM_02. Origin is CAM_01, Destination is CAM_02! (5 vehicles)
  // V09..V10 travel CAM_01 -> CAM_03. Origin is CAM_01, Destination is CAM_03! (2 vehicles)
  // Total CAM_01 -> CAM_03 = 3 + 2 = 5 vehicles!
  // Total CAM_01 -> CAM_02 = 5 vehicles!
  // Intermediate CAM_02 is NOT counted as destination for V01..V03!
  assert(od12 !== undefined, 'OD pair CAM_01->CAM_02 should exist');
  assert.strictEqual(od12.vehicleCount, 5, `Expected 5 for CAM_01->CAM_02, got ${od12.vehicleCount}`);

  assert(od13 !== undefined, 'OD pair CAM_01->CAM_03 should exist');
  assert.strictEqual(od13.vehicleCount, 5, `Expected 5 for CAM_01->CAM_03, got ${od13.vehicleCount}`);

  // No journeys originated at CAM_02 that terminated at CAM_03 (V01..V03 originated at CAM_01)
  assert.strictEqual(od23, undefined, 'CAM_02->CAM_03 should have 0 journeys originating at CAM_02');
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Road-Aware Speed Analytics
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 5. Road-Aware Speed Analytics ---');

const speedRes = speedAnalytics.calculateSpeedAnalytics(detections);

runTest('Speed is computed using road distance over travel time for valid segments', () => {
  const r12Speed = speedRes.routes.find((r) => r.route === 'CAM_01->CAM_02');
  assert(r12Speed !== undefined);
  // Road distance is 2.75 km. Travel time is 360 seconds (6 minutes).
  // Speed = 2.75 / (360/3600) = 27.5 km/h
  assert.strictEqual(r12Speed.averageSpeedKmh, 27.5, `Expected 27.5 km/h, got ${r12Speed.averageSpeedKmh}`);
  assert.strictEqual(r12Speed.averageTravelTimeSeconds, 360);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. GIS Heatmap Intensities
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 6. GIS Heatmap Intensities ---');

runTest('Heatmap intensity scales proportionally to traffic metrics with no random values', () => {
  const heatPoints = normalizeHeatmapData(densityRes.cameras, 'vehicleCount');
  assert(heatPoints.length >= 3, `Expected at least 3 heat points, got ${heatPoints.length}`);

  // Find intensities by lat/lng
  const p1 = heatPoints.find((p) => Math.abs(p[0] - 13.0827) < 0.001);
  const p2 = heatPoints.find((p) => Math.abs(p[0] - 13.0731) < 0.001);
  const p3 = heatPoints.find((p) => Math.abs(p[0] - 13.0878) < 0.001);

  assert.strictEqual(p2[2], 1.0, `CAM_02 (max=20) should have intensity 1.0, got ${p2[2]}`);
  assert.strictEqual(p1[2], 0.5, `CAM_01 (10/20) should have intensity 0.5, got ${p1[2]}`);
  assert.strictEqual(p3[2], 0.25, `CAM_03 (5/20) should have intensity 0.25, got ${p3[2]}`);
});

console.log('\n======================================================================');
console.log(`🏁 GROUND-TRUTH TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
console.log('======================================================================\n');

if (passedTests !== totalTests) {
  process.exit(1);
}
