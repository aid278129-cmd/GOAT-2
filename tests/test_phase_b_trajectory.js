'use strict';

/**
 * Phase B - Trajectory Completion Comprehensive Unit & Integration Test Suite
 * Tests all requirements from Phase B Roadmap specification.
 */

const assert = require('assert');
const { haversineDistance, calculateEstimatedSpeed } = require('../services/geoService');
const trajectoryService = require('../services/trajectoryService');

console.log('======================================================================');
console.log('🧪 RUNNING PHASE B TRAJECTORY COMPLETION TEST SUITE');
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
// 1. GeoService Unit Tests
// ─────────────────────────────────────────────────────────────────────────────
console.log('--- 1. GeoService Unit Tests ---');

runTest('haversineDistance returns 0 for identical coordinates', () => {
  const d = haversineDistance(13.0827, 80.2707, 13.0827, 80.2707);
  assert.strictEqual(d, 0);
});

runTest('haversineDistance calculates accurate distance between Chennai Central & T. Nagar (~1.5 - 2.5 km)', () => {
  // Junction A (Chennai Central): 13.0827, 80.2707
  // Junction B (T. Nagar): 13.0731, 80.2609
  const d = haversineDistance(13.0827, 80.2707, 13.0731, 80.2609);
  assert(d !== null, 'Distance should not be null');
  assert(d > 1.0 && d < 2.5, `Distance was ${d} km, expected ~1.5 - 2.0 km`);
});

runTest('haversineDistance returns null for invalid coordinates', () => {
  assert.strictEqual(haversineDistance(null, 80, 13, 80), null);
  assert.strictEqual(haversineDistance(13, undefined, 13, 80), null);
  assert.strictEqual(haversineDistance('invalid', 80, 13, 80), null);
  assert.strictEqual(haversineDistance(95, 80, 13, 80), null); // Out of range lat
  assert.strictEqual(haversineDistance(13, -190, 13, 80), null); // Out of range lon
});

runTest('calculateEstimatedSpeed correctly computes km/h', () => {
  // 4 km in 8 minutes (480 seconds) = 30 km/h
  const speed = calculateEstimatedSpeed(4.0, 480);
  assert(speed !== null, 'Speed should not be null');
  assert(Math.abs(speed - 30.0) < 0.01, `Speed was ${speed}, expected 30.0`);
});

runTest('calculateEstimatedSpeed returns null for invalid travel time or distance', () => {
  assert.strictEqual(calculateEstimatedSpeed(4.0, 0), null);
  assert.strictEqual(calculateEstimatedSpeed(4.0, -10), null);
  assert.strictEqual(calculateEstimatedSpeed(-4.0, 480), null);
  assert.strictEqual(calculateEstimatedSpeed(null, 480), null);
  assert.strictEqual(calculateEstimatedSpeed(4.0, null), null);
  assert.strictEqual(calculateEstimatedSpeed(NaN, 480), null);
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. TrajectoryService: 3-Camera Golden Path
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. TrajectoryService 3-Camera Golden Path ---');

const mockDetections = [
  // Out of order timestamps to test sorting!
  {
    id: 'det-3',
    plate: 'TN45AB1234',
    cameraId: 'CAM_03',
    cameraName: 'Junction C',
    cameraLocation: 'Perambur',
    timestamp: '2026-09-25T10:15:00Z',
    latitude: 13.0878,
    longitude: 80.2785,
    confidence: 0.95,
  },
  {
    id: 'det-1',
    plate: 'TN45AB1234',
    cameraId: 'CAM_01',
    cameraName: 'Junction A',
    cameraLocation: 'Chennai Central',
    timestamp: '2026-09-25T10:00:00Z',
    latitude: 13.0827,
    longitude: 80.2707,
    confidence: 0.96,
  },
  {
    id: 'det-2',
    plate: 'TN45AB1234',
    cameraId: 'CAM_02',
    cameraName: 'Junction B',
    cameraLocation: 'T. Nagar',
    timestamp: '2026-09-25T10:08:00Z',
    latitude: 13.0731,
    longitude: 80.2609,
    confidence: 0.94,
  },
];

runTest('Correctly sorts sightings chronologically regardless of input order', () => {
  const sightings = trajectoryService.getPlateSightings('TN45AB1234', mockDetections);
  assert.strictEqual(sightings.length, 3);
  assert.strictEqual(sightings[0].cameraId, 'CAM_01');
  assert.strictEqual(sightings[1].cameraId, 'CAM_02');
  assert.strictEqual(sightings[2].cameraId, 'CAM_03');
});

runTest('buildTrajectory calculates accurate segments, times, distances, and speeds', () => {
  const traj = trajectoryService.buildTrajectory('TN45AB1234', mockDetections);
  assert.strictEqual(traj.found, true);
  assert.strictEqual(traj.totalSightings, 3);
  assert.strictEqual(traj.points.length, 3);
  assert.strictEqual(traj.segments.length, 2);

  // Segment 1: CAM_01 -> CAM_02 (10:00 to 10:08 -> 480 seconds = 8 minutes)
  const seg1 = traj.segments[0];
  assert.strictEqual(seg1.fromCameraId, 'CAM_01');
  assert.strictEqual(seg1.toCameraId, 'CAM_02');
  assert.strictEqual(seg1.travelTimeSeconds, 480);
  assert.strictEqual(seg1.travelMins, 8);
  assert(seg1.distanceKm > 1.2 && seg1.distanceKm < 2.0, `Seg 1 distance was ${seg1.distanceKm}`);
  assert(seg1.estimatedSpeedKmh > 5 && seg1.estimatedSpeedKmh < 30, `Seg 1 speed was ${seg1.estimatedSpeedKmh}`);
  assert.strictEqual(seg1.validForAnalytics, true);

  // Segment 2: CAM_02 -> CAM_03 (10:08 to 10:15 -> 420 seconds = 7 minutes)
  const seg2 = traj.segments[1];
  assert.strictEqual(seg2.fromCameraId, 'CAM_02');
  assert.strictEqual(seg2.toCameraId, 'CAM_03');
  assert.strictEqual(seg2.travelTimeSeconds, 420);
  assert.strictEqual(seg2.travelMins, 7);
  assert(seg2.distanceKm > 1.5 && seg2.distanceKm < 3.5, `Seg 2 distance was ${seg2.distanceKm}`);
  assert(seg2.estimatedSpeedKmh > 10 && seg2.estimatedSpeedKmh < 40, `Seg 2 speed was ${seg2.estimatedSpeedKmh}`);
  assert.strictEqual(seg2.validForAnalytics, true);

  // Summary Metrics
  assert(traj.totalDistanceKm > 2.5 && traj.totalDistanceKm < 5.5, `Total distance was ${traj.totalDistanceKm}`);
  assert.strictEqual(traj.totalTravelTimeSeconds, 900); // 15 minutes
  assert.strictEqual(traj.totalTravelTimeMinutes, 15.0);
  assert(traj.averageJourneySpeedKmh > 10 && traj.averageJourneySpeedKmh < 30, `Avg speed was ${traj.averageJourneySpeedKmh}`);
  assert.strictEqual(traj.validSegmentCount, 2);
  assert.strictEqual(traj.invalidSegmentCount, 0);

  // Backward compatibility check
  assert(Array.isArray(traj.trail), 'Must have trail array');
  assert.strictEqual(traj.trail.length, 3);
  assert(traj.totalMinutes !== undefined, 'Must have totalMinutes');
  assert(Array.isArray(traj.cameras), 'Must have cameras array');
  assert.strictEqual(traj.cameras.length, 3);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Edge Cases & Anomaly Validation
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Edge Cases & Anomaly Validation ---');

runTest('Handles impossible speed (> 180 km/h) by flagging IMPOSSIBLE_TRAVEL anomaly', () => {
  // Same distance (~1.5km), but travel time is 5 seconds! Speed ~ 1000+ km/h
  const anomalyDetections = [
    {
      id: 'd1',
      plate: 'FAST999',
      cameraId: 'CAM_01',
      timestamp: '2026-09-25T10:00:00Z',
      latitude: 13.0827,
      longitude: 80.2707,
    },
    {
      id: 'd2',
      plate: 'FAST999',
      cameraId: 'CAM_02',
      timestamp: '2026-09-25T10:00:05Z', // 5 seconds later!
      latitude: 13.0731,
      longitude: 80.2609,
    },
  ];

  const traj = trajectoryService.buildTrajectory('FAST999', anomalyDetections);
  assert.strictEqual(traj.segments.length, 1);
  const seg = traj.segments[0];
  assert.strictEqual(seg.validForAnalytics, false);
  assert(seg.anomaly !== null, 'Should have anomaly object');
  assert.strictEqual(seg.anomaly.type, 'IMPOSSIBLE_TRAVEL');
  assert(seg.estimatedSpeedKmh > 180, 'Speed should exceed 180 km/h');
  assert.strictEqual(traj.validSegmentCount, 0);
  assert.strictEqual(traj.invalidSegmentCount, 1);
});

runTest('Handles same-camera repeated sightings without calculating false movement', () => {
  const sameCamDetections = [
    {
      id: 'd1',
      plate: 'SAME001',
      cameraId: 'CAM_01',
      timestamp: '2026-09-25T10:00:00Z',
      latitude: 13.0827,
      longitude: 80.2707,
    },
    {
      id: 'd2',
      plate: 'SAME001',
      cameraId: 'CAM_01', // Repeated same camera!
      timestamp: '2026-09-25T10:00:30Z',
      latitude: 13.0827,
      longitude: 80.2707,
    },
  ];

  const traj = trajectoryService.buildTrajectory('SAME001', sameCamDetections);
  assert.strictEqual(traj.totalSightings, 2);
  const seg = traj.segments[0];
  assert.strictEqual(seg.isSameCamera, true);
  assert.strictEqual(seg.validForAnalytics, false);
  assert(seg.validationIssues.includes('SAME_CAMERA_STATIONARY'));
});

runTest('Handles missing or (0,0) coordinates gracefully without crashing', () => {
  const badCoordDetections = [
    {
      id: 'd1',
      plate: 'NOCOORD',
      cameraId: 'CAM_01',
      timestamp: '2026-09-25T10:00:00Z',
      latitude: null,
      longitude: null,
    },
    {
      id: 'd2',
      plate: 'NOCOORD',
      cameraId: 'CAM_02',
      timestamp: '2026-09-25T10:05:00Z',
      lat: 0,
      lng: 0,
    },
  ];

  const traj = trajectoryService.buildTrajectory('NOCOORD', badCoordDetections);
  assert.strictEqual(traj.found, true);
  assert.strictEqual(traj.segments.length, 1);
  const seg = traj.segments[0];
  assert.strictEqual(seg.validForAnalytics, false);
  assert(seg.validationIssues.includes('MISSING_COORDINATES'));
  assert.strictEqual(seg.distanceKm, null);
  assert.strictEqual(seg.estimatedSpeedKmh, null);
});

runTest('Handles malformed timestamps gracefully without crashing', () => {
  const badTimeDetections = [
    {
      id: 'd1',
      plate: 'BADTIME',
      cameraId: 'CAM_01',
      timestamp: 'not-a-valid-date',
      latitude: 13.0827,
      longitude: 80.2707,
    },
    {
      id: 'd2',
      plate: 'BADTIME',
      cameraId: 'CAM_02',
      timestamp: '2026-09-25T10:05:00Z',
      latitude: 13.0731,
      longitude: 80.2609,
    },
  ];

  const traj = trajectoryService.buildTrajectory('BADTIME', badTimeDetections);
  assert.strictEqual(traj.found, true);
  const seg = traj.segments[0];
  assert.strictEqual(seg.validForAnalytics, false);
  assert(seg.validationIssues.includes('MALFORMED_TIMESTAMP'));
});

runTest('Handles single sighting cleanly (0 segments)', () => {
  const singleDetections = [
    {
      id: 'd1',
      plate: 'ONEPOINT',
      cameraId: 'CAM_01',
      timestamp: '2026-09-25T10:00:00Z',
      latitude: 13.0827,
      longitude: 80.2707,
    },
  ];

  const traj = trajectoryService.buildTrajectory('ONEPOINT', singleDetections);
  assert.strictEqual(traj.found, true);
  assert.strictEqual(traj.totalSightings, 1);
  assert.strictEqual(traj.segments.length, 0);
  assert.strictEqual(traj.totalDistanceKm, 0);
  assert.strictEqual(traj.averageJourneySpeedKmh, null);
});

runTest('Handles plate not found cleanly without throwing error', () => {
  const traj = trajectoryService.buildTrajectory('NOTFOUND99', mockDetections);
  assert.strictEqual(traj.found, false);
  assert.strictEqual(traj.success, false);
  assert.strictEqual(traj.totalSightings, 0);
  assert(Array.isArray(traj.trail) && traj.trail.length === 0);
  assert(Array.isArray(traj.points) && traj.points.length === 0);
});

console.log('\n======================================================================');
console.log(`🏁 TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
console.log('======================================================================');

if (passedTests !== totalTests) {
  process.exit(1);
}
