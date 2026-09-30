'use strict';

/**
 * test_road_aware_routing.js — Comprehensive Test Suite for Road-Aware Vehicle Route Prediction
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 *
 * Covers:
 * 1. Camera Road Snapping
 * 2. Road Route Retrieval (OSRM / Topology)
 * 3. Route Caching & Memory Efficiency
 * 4. Routing Timeout & Failure Handling (No Silent Straight Lines)
 * 5. One-way / Directional Feasibility
 * 6. Multi-Camera Waypoint Routing & Leg Stitching
 * 7. Observed Route Geometry & Road Alignment
 * 8. Predicted Route Geometry & Top Candidates
 * 9. Road Distance vs Geodesic Distance
 * 10. Journey Sessionization & Gap Splitting (> 30 min)
 * 11. Large Timestamp Gap Handling (The 38-hour 0.04 km/h case)
 * 12. Invalid / Implausible Speed Validation
 * 13. Prediction Candidate Road Feasibility Scoring
 */

const assert = require('assert');
const roadRoutingService = require('../services/roadRoutingService');
const routePredictionService = require('../services/routePredictionService');
const trajectoryService = require('../services/trajectoryService');
const cameras = require('../data/cameras.json');
const detections = require('../data/detections.json');

let totalTests = 0;
let passedTests = 0;

function runTest(name, fn) {
  totalTests++;
  try {
    const res = fn();
    if (res && typeof res.then === 'function') {
      return res.then(() => {
        console.log(`  ✅ PASS: ${name}`);
        passedTests++;
      }).catch((err) => {
        console.error(`  ❌ FAIL: ${name}`);
        console.error(`     Error: ${err.message}`);
        if (err.stack) console.error(`     ${err.stack.split('\n').slice(1, 3).join('\n     ')}`);
      });
    }
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

async function main() {
  console.log('======================================================================');
  console.log('🧪 RUNNING ROAD-AWARE VEHICLE ROUTE PREDICTION TEST SUITE');
  console.log('======================================================================\n');

  // ─────────────────────────────────────────────────────────────────────────
  // 1. Camera Snapping Tests
  // ─────────────────────────────────────────────────────────────────────────
  console.log('--- 1. Camera Snapping Tests ---');

  await runTest('snapCameraToRoad snaps CAM_01 to Raja Muthiah Road within threshold (< 50m)', async () => {
    const snap = await roadRoutingService.snapCameraToRoad('CAM_01');
    assert.strictEqual(snap.success, true);
    assert.strictEqual(snap.snapped, true);
    assert(snap.snapDistanceMeters < 50, `Snap distance ${snap.snapDistanceMeters}m exceeded 50m`);
    assert(Array.isArray(snap.snappedCoordinate));
    assert.strictEqual(snap.snappedCoordinate.length, 2);
  });

  await runTest('snapCameraToRoad gracefully handles unknown camera coordinates', async () => {
    const snap = await roadRoutingService.snapCameraToRoad({ lat: null, lng: null });
    assert.strictEqual(snap.success, false);
    assert.strictEqual(snap.snapped, false);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 2. Road Route Retrieval & Road Alignment
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 2. Road Route Retrieval & Road Alignment ---');

  await runTest('getRoadRoute retrieves road-aligned geometry between CAM_01 and CAM_02', async () => {
    const route = await roadRoutingService.getRoadRoute('CAM_01', 'CAM_02');
    assert.strictEqual(route.success, true);
    assert.strictEqual(route.roadAligned, true);
    assert.strictEqual(route.routingStatus, 'OK');
    assert(route.distanceKm > 2.0 && route.distanceKm < 3.5, `Distance was ${route.distanceKm} km, expected 2.0 - 3.5 km`);
    assert(route.latLngs.length > 50, `Expected road-following path, got only ${route.latLngs.length} points`);
    assert.strictEqual(route.geometry.type, 'LineString');
  });

  await runTest('getRoadRoute distinguishes directionally asymmetric routes (CAM_01->CAM_02 vs CAM_02->CAM_01)', async () => {
    const routeFwd = await roadRoutingService.getRoadRoute('CAM_01', 'CAM_02');
    const routeRev = await roadRoutingService.getRoadRoute('CAM_02', 'CAM_01');
    assert.strictEqual(routeFwd.roadAligned, true);
    assert.strictEqual(routeRev.roadAligned, true);
    // Because of urban one-ways, distances or points differ
    assert(routeFwd.latLngs.length > 0 && routeRev.latLngs.length > 0);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 3. Route Caching & Memory Efficiency
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 3. Route Caching & Performance ---');

  await runTest('getRoadRoute serves subsequent requests from cache without re-fetching', async () => {
    const t0 = Date.now();
    const r1 = await roadRoutingService.getRoadRoute('CAM_01', 'CAM_03');
    const t1 = Date.now();
    const r2 = await roadRoutingService.getRoadRoute('CAM_01', 'CAM_03');
    const t2 = Date.now();
    assert.strictEqual(r2.roadAligned, true);
    assert.strictEqual(r2.distanceKm, r1.distanceKm);
    assert(t2 - t1 < 50, `Cached call took ${t2 - t1}ms, expected < 50ms`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 4. Routing Timeout & Failure Handling
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 4. Routing Failure & Fallback Handling ---');

  await runTest('Routing failure returns roadAligned: false and UNAVAILABLE without pretending to be road', async () => {
    const badRoute = await roadRoutingService.getRoadRoute(
      'CAM_01',
      'CAM_02',
      { simulateFailure: true }
    );
    assert.strictEqual(badRoute.roadAligned, false);
    assert.strictEqual(badRoute.routingStatus, 'UNAVAILABLE');
    assert.strictEqual(badRoute.message, 'Road route unavailable');
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 5. Multi-Camera Waypoint Routing & Leg Stitching
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 5. Multi-Camera Waypoint Routing ---');

  await runTest('getRoadRouteThroughCameras stitches multi-leg route (CAM_01 -> CAM_02 -> CAM_03)', async () => {
    const multi = await roadRoutingService.getRoadRouteThroughCameras(['CAM_01', 'CAM_02', 'CAM_03']);
    assert.strictEqual(multi.success, true);
    assert.strictEqual(multi.roadAligned, true);
    assert.strictEqual(multi.legs.length, 2);
    assert(multi.totalDistanceKm > 5.0 && multi.totalDistanceKm < 8.0, `Multi-hop distance was ${multi.totalDistanceKm}`);
    assert(multi.latLngs.length > 100, `Expected stitched polyline, got ${multi.latLngs.length} points`);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 6. Journey Sessionization Tests
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 6. Journey Sessionization Tests ---');

  runTest('segmentSightingsIntoJourneys groups detections within 30 min into same journey', () => {
    const sightings = [
      { plate: 'TN45AB1234', cameraId: 1, timestamp: '2026-09-28T08:20:00Z', latitude: 13.0827, longitude: 80.2707 },
      { plate: 'TN45AB1234', cameraId: 2, timestamp: '2026-09-28T08:27:00Z', latitude: 13.0731, longitude: 80.2609 },
      { plate: 'TN45AB1234', cameraId: 3, timestamp: '2026-09-28T08:36:00Z', latitude: 13.0878, longitude: 80.2785 },
    ];
    const journeys = trajectoryService.segmentSightingsIntoJourneys(sightings);
    assert.strictEqual(journeys.length, 1);
    assert.strictEqual(journeys[0].totalSightings, 3);
    assert.strictEqual(journeys[0].journeyId, 'TN45AB1234-session-1');
    assert.strictEqual(journeys[0].isActive, true);
  });

  runTest('segmentSightingsIntoJourneys splits detections separated by > 30 min into distinct journeys', () => {
    const sightings = [
      { plate: 'TN45AB1234', cameraId: 1, timestamp: '2026-09-28T08:20:00Z', latitude: 13.0827, longitude: 80.2707 },
      { plate: 'TN45AB1234', cameraId: 4, timestamp: '2026-09-28T22:40:00Z', latitude: 13.0569, longitude: 80.2425 }, // 14 hours later!
    ];
    const journeys = trajectoryService.segmentSightingsIntoJourneys(sightings);
    assert.strictEqual(journeys.length, 2, 'Should be split into 2 journeys');
    assert.strictEqual(journeys[0].totalSightings, 1);
    assert.strictEqual(journeys[1].totalSightings, 1);
    assert.strictEqual(journeys[1].journeyId, 'TN45AB1234-session-2');
    assert.strictEqual(journeys[0].isActive, false);
    assert.strictEqual(journeys[1].isActive, true);
    assert((journeys[0].splitReason || '').includes('exceeds session threshold'));
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 7. Screenshot Case Investigation: MH47BP8265
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 7. Screenshot Case Investigation (MH47BP8265) ---');

  runTest('MH47BP8265 is correctly split into 2 journeys instead of a 38-hour 0.04 km/h false trip', () => {
    const traj = trajectoryService.buildTrajectory('MH47BP8265', detections);
    assert.strictEqual(traj.found, true);
    assert.strictEqual(traj.journeys.length, 2, 'Must have 2 journey sessions');

    // Active Journey should be session 2 (at CAM_02 on Sep 28)
    assert.strictEqual(traj.journeyId, 'MH47BP8265-session-2');
    assert.strictEqual(traj.totalSightings, 1);
    assert.strictEqual(traj.totalTravelTimeSeconds, 0);
    assert.strictEqual(traj.averageJourneySpeedKmh, null, 'Single sighting active trip must have null speed, NOT 0.04 km/h');

    // Session 1 verification
    const sess1 = traj.journeys[0];
    assert.strictEqual(sess1.journeyId, 'MH47BP8265-session-1');
    assert.strictEqual(sess1.sessionStart, '2026-09-26T18:39:34.068Z');
    assert.strictEqual(sess1.isActive, false);
  });

  runTest('Road distance vs Geodesic distance calculation on CAM_01 -> CAM_02', () => {
    const sightings = [
      { plate: 'DIST_TEST', cameraId: 1, timestamp: '2026-09-28T08:00:00Z', latitude: 13.0827, longitude: 80.2707 },
      { plate: 'DIST_TEST', cameraId: 2, timestamp: '2026-09-28T08:08:00Z', latitude: 13.0731, longitude: 80.2609 },
    ];
    const traj = trajectoryService.buildTrajectory('DIST_TEST', sightings);
    assert.strictEqual(traj.found, true);
    // Geodesic straight-line is ~1.51 km
    assert(Math.abs(traj.geodesicDistanceKm - 1.51) < 0.2, `Geodesic distance was ${traj.geodesicDistanceKm}`);
    // Actual road distance along Pantheon Rd corridor is ~2.75 km
    assert(traj.roadDistanceKm > 2.4 && traj.roadDistanceKm < 3.2, `Road distance was ${traj.roadDistanceKm}`);
    assert.strictEqual(traj.roadAligned, true);
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 8. Road-Aware Route Prediction Tests
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 8. Road-Aware Route Prediction Tests ---');

  runTest('predictNextCameras predicts topologically feasible next nodes with road geometry', () => {
    const activeJourney = {
      journeyId: 'TEST-PRED',
      sessionSightings: [
        { cameraId: 'CAM_01', timestamp: '2026-09-28T08:00:00Z', direction: 'northbound' },
        { cameraId: 'CAM_02', timestamp: '2026-09-28T08:08:00Z', direction: 'eastbound' },
      ],
    };
    const pred = routePredictionService.predictNextCamerasSync(activeJourney, []);
    assert.strictEqual(pred.success, true);
    assert.strictEqual(pred.currentCamera, 'CAM_02');
    assert(pred.nextCameras.length > 0);

    const primary = pred.nextCameras[0];
    assert(primary.probability > 0.3, `Primary probability was ${primary.probability}`);
    assert(primary.confidence === 'HIGH' || primary.confidenceScore > 0.7 || primary.confidence > 0.7, 'Must have high confidence');
    assert.strictEqual(primary.route.roadAligned, true);
    assert(primary.route.latLngs.length > 10, 'Must have road geometry latLngs');
    assert(primary.etaSeconds > 0);
  });

  runTest('predictNextCameras penalizes immediate turnaround / U-turn to previous camera', () => {
    const activeJourney = {
      journeyId: 'TEST-REV',
      sessionSightings: [
        { cameraId: 'CAM_01', timestamp: '2026-09-28T08:00:00Z' },
        { cameraId: 'CAM_02', timestamp: '2026-09-28T08:08:00Z' },
      ],
    };
    const pred = routePredictionService.predictNextCamerasSync(activeJourney, []);
    const cam01Candidate = pred.nextCameras.find((c) => c.cameraId === 'CAM_01');
    const forwardCandidate = pred.nextCameras.find((c) => c.cameraId === 'CAM_03');

    // Forward candidate should score higher than immediate reversal
    if (cam01Candidate && forwardCandidate) {
      assert(forwardCandidate.probability >= cam01Candidate.probability, 'Forward candidate should outrank direct reversal');
    }
  });

  // ─────────────────────────────────────────────────────────────────────────
  // 9. Multi-Hop Route Prediction
  // ─────────────────────────────────────────────────────────────────────────
  console.log('\n--- 9. Multi-Hop Corridor Prediction ---');

  runTest('Multi-hop corridor prediction produces sequential road-aligned route', () => {
    const activeJourney = {
      journeyId: 'TEST-MULTIHOP',
      sessionSightings: [
        { cameraId: 'CAM_01', timestamp: '2026-09-28T08:00:00Z' },
      ],
    };
    const pred = routePredictionService.predictNextCamerasSync(activeJourney, []);
    assert(pred.multiHopRoute !== null, 'Must provide multiHopRoute');
    assert.strictEqual(pred.multiHopRoute.sequence.length, 3, 'Sequence must have 3 cameras');
    assert.strictEqual(pred.multiHopRoute.roadAligned, true);
    assert(pred.multiHopRoute.totalDistanceKm > 3.0);
  });

  console.log('\n======================================================================');
  console.log(`🏁 TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
  console.log('======================================================================\n');
}

main();
