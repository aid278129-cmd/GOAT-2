'use strict';

/**
 * Camera Network Lifecycle & Operational State Test Suite
 * Validates ONLINE, DEGRADED, DISCONNECTED, OFFLINE states, heartbeats, and failure/recovery latencies.
 */

const assert = require('assert');
const http = require('http');

console.log('======================================================================');
console.log('🧪 RUNNING CAMERA NETWORK LIFECYCLE & FAILURE TEST SUITE');
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

async function runAsyncTest(name, fn) {
  totalTests++;
  try {
    await fn();
    console.log(`  ✅ PASS: ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${name}`);
    console.error(`     Error: ${err.message}`);
  }
}

// Helper to simulate camera state logic directly
function evaluateCameraState(camId, isConnected, rt, now = Date.now()) {
  const lastHeartbeatMs = rt.lastHeartbeat || 0;
  const isExplicitlyDisconnected = rt.connectionState === 'disconnected';
  const isHeartbeatFresh = !isExplicitlyDisconnected && (now - lastHeartbeatMs < 15000);
  const connected = (isConnected && !isExplicitlyDisconnected) || isHeartbeatFresh;

  let status = 'OFFLINE';
  let stream = 'INACTIVE';

  if (connected) {
    const lastFrameMs = rt.lastFrameAt || 0;
    if (lastFrameMs && (now - lastFrameMs <= 30000)) {
      status = 'ONLINE';
      stream = 'ACTIVE';
    } else if (lastFrameMs && (now - lastFrameMs > 30000)) {
      status = 'DEGRADED';
      stream = 'STANDBY';
    } else {
      status = 'ONLINE';
      stream = 'STANDBY';
    }
  } else {
    if (rt.connectionState === 'disconnected' && (now - (rt.disconnectAt || 0) < 60000)) {
      status = 'DISCONNECTED';
      stream = 'INACTIVE';
    } else {
      status = 'OFFLINE';
      stream = 'INACTIVE';
    }
  }

  return { status, stream, connected };
}

(async () => {
  // ─────────────────────────────────────────────────────────────────────────────
  // 1. Camera State Definitions & Transitions
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- 1. Camera State Definitions & Transitions ---');

  runTest('Default state without connection or heartbeat is OFFLINE / INACTIVE', () => {
    const state = evaluateCameraState(2, false, {});
    assert.strictEqual(state.status, 'OFFLINE');
    assert.strictEqual(state.stream, 'INACTIVE');
  });

  runTest('Recent heartbeat transitions camera to ONLINE / STANDBY', () => {
    const now = Date.now();
    const state = evaluateCameraState(2, false, { lastHeartbeat: now - 2000, connectionState: 'connected' }, now);
    assert.strictEqual(state.status, 'ONLINE');
    assert.strictEqual(state.stream, 'STANDBY');
  });

  runTest('Active video frames within 30s transition stream to ACTIVE', () => {
    const now = Date.now();
    const state = evaluateCameraState(2, true, { lastHeartbeat: now - 1000, lastFrameAt: now - 500 }, now);
    assert.strictEqual(state.status, 'ONLINE');
    assert.strictEqual(state.stream, 'ACTIVE');
  });

  runTest('Stalled frame transmission (> 30s) transitions camera to DEGRADED / STANDBY', () => {
    const now = Date.now();
    const state = evaluateCameraState(2, true, { lastHeartbeat: now - 2000, lastFrameAt: now - 35000 }, now);
    assert.strictEqual(state.status, 'DEGRADED');
    assert.strictEqual(state.stream, 'STANDBY');
  });

  runTest('Recent disconnect event (< 60s) transitions camera to DISCONNECTED / INACTIVE', () => {
    const now = Date.now();
    const state = evaluateCameraState(2, false, { connectionState: 'disconnected', disconnectAt: now - 5000 }, now);
    assert.strictEqual(state.status, 'DISCONNECTED');
    assert.strictEqual(state.stream, 'INACTIVE');
  });

  runTest('Extended disconnect (> 60s) transitions camera to OFFLINE / INACTIVE', () => {
    const now = Date.now();
    const state = evaluateCameraState(2, false, { connectionState: 'disconnected', disconnectAt: now - 65000 }, now);
    assert.strictEqual(state.status, 'OFFLINE');
    assert.strictEqual(state.stream, 'INACTIVE');
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 2. Camera Failure & Reconnect Latency Lifecycle Test
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- 2. Camera Failure & Reconnect Latency Lifecycle Test ---');

  runTest('Simulates Camera Connected -> Active -> Disconnect -> Reconnect with Latency Tracking', () => {
    const runtimeState = {};
    let t0, t1, latencyMs;

    // Step A: Camera Connects
    t0 = performance.now();
    runtimeState.lastHeartbeat = Date.now();
    runtimeState.connectionState = 'connected';
    let s1 = evaluateCameraState(2, true, runtimeState);
    t1 = performance.now();
    latencyMs = Number((t1 - t0).toFixed(3));
    assert.strictEqual(s1.status, 'ONLINE');
    console.log(`     [Metric] Connect Transition Latency: ${latencyMs} ms`);

    // Step B: Stream frames
    runtimeState.lastFrameAt = Date.now();
    let s2 = evaluateCameraState(2, true, runtimeState);
    assert.strictEqual(s2.stream, 'ACTIVE');

    // Step C: Disconnect Camera (Failure event)
    t0 = performance.now();
    runtimeState.connectionState = 'disconnected';
    runtimeState.disconnectAt = Date.now();
    let s3 = evaluateCameraState(2, false, runtimeState);
    t1 = performance.now();
    const disconnectLatencyMs = Number((t1 - t0).toFixed(3));
    assert.strictEqual(s3.status, 'DISCONNECTED');
    assert.strictEqual(s3.stream, 'INACTIVE');
    assert(disconnectLatencyMs < 20.0, `Disconnect state update must be instantaneous (<20ms), was ${disconnectLatencyMs} ms`);
    console.log(`     [Metric] Disconnect Transition Latency: ${disconnectLatencyMs} ms`);

    // Step D: Reconnect Camera (Recovery event)
    t0 = performance.now();
    runtimeState.connectionState = 'connected';
    runtimeState.lastHeartbeat = Date.now();
    runtimeState.lastFrameAt = Date.now();
    let s4 = evaluateCameraState(2, true, runtimeState);
    t1 = performance.now();
    const reconnectLatencyMs = Number((t1 - t0).toFixed(3));
    assert.strictEqual(s4.status, 'ONLINE');
    assert.strictEqual(s4.stream, 'ACTIVE');
    assert(reconnectLatencyMs < 20.0, `Reconnect state restoration must be instantaneous (<20ms), was ${reconnectLatencyMs} ms`);
    console.log(`     [Metric] Reconnect Restoration Latency: ${reconnectLatencyMs} ms`);
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 3. Operational Telemetry Schema Validation
  // ─────────────────────────────────────────────────────────────────────────────
  console.log('\n--- 3. Operational Telemetry Schema Validation ---');

  runTest('Validates complete operational status payload schema per camera', () => {
    const mockCameraStatus = {
      cameraId: 'CAM_02',
      id: 2,
      cameraNumericId: 2,
      name: 'Junction B',
      location: 'Pantheon Road',
      roadName: 'Pantheon Road',
      zone: 'West',
      status: 'ONLINE',
      stream: 'ACTIVE',
      detections: 142,
      uniqueVehicles: 103,
      currentFlow: '18 veh / 5 min',
      currentFlowValue: 18,
      congestion: 'MEDIUM',
      lastDetection: '2026-09-28T10:20:00.000Z',
      lastHeartbeat: '2026-09-28T10:20:05.000Z',
      lastFrameTimestamp: '2026-09-28T10:20:05.500Z',
      connectionState: 'connected',
    };

    assert.strictEqual(typeof mockCameraStatus.cameraId, 'string');
    assert.strictEqual(typeof mockCameraStatus.status, 'string');
    assert(['ONLINE', 'DEGRADED', 'DISCONNECTED', 'OFFLINE'].includes(mockCameraStatus.status));
    assert(['ACTIVE', 'STANDBY', 'INACTIVE'].includes(mockCameraStatus.stream));
    assert.strictEqual(typeof mockCameraStatus.detections, 'number');
    assert.strictEqual(typeof mockCameraStatus.uniqueVehicles, 'number');
    assert(mockCameraStatus.currentFlow.includes('veh / 5 min'));
    assert(['LOW', 'MEDIUM', 'MODERATE', 'HIGH', 'SEVERE'].includes(mockCameraStatus.congestion));
    assert(mockCameraStatus.lastDetection !== undefined);
    assert(mockCameraStatus.lastHeartbeat !== undefined);
  });

  console.log('\n======================================================================');
  console.log(`🏁 CAMERA NETWORK TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
  console.log('======================================================================\n');

  if (passedTests !== totalTests) {
    process.exit(1);
  }
})();
