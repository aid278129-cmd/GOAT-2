'use strict';

/**
 * Watchlist & Real-Time Alert System Test Suite
 * Validates Watchlist CRUD, real-time alert generation, alert schema,
 * duplicate alert suppression, and cross-camera tracking integration.
 */

const assert = require('assert');
const crypto = require('crypto');

console.log('======================================================================');
console.log('🧪 RUNNING WATCHLIST & REAL-TIME ALERT TEST SUITE');
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
// Simulated Watchlist & Alert Engine (mirroring server.js implementation)
// ─────────────────────────────────────────────────────────────────────────────
const watchlist = new Map();
const alertHistory = [];
const lastAlertTime = new Map();
const ALERT_COOLDOWN_MS = 60000;

function addWatchlistEntry(plate, reason, priority = 'HIGH', notes = '') {
  const norm = plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
  const entry = {
    plate: norm,
    reason,
    priority,
    notes,
    active: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  watchlist.set(norm, entry);
  return entry;
}

function processDetectionAlert(detection, options = {}) {
  const normPlate = (detection.plate || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  if (!watchlist.has(normPlate)) return null;

  const wlEntry = watchlist.get(normPlate);
  if (!wlEntry || wlEntry.active === false) return null;

  const now = options.now || Date.now();
  const alertKey = `${normPlate}:${detection.cameraId}`;
  const lastAlertTs = lastAlertTime.get(alertKey) || 0;
  const isCooldownActive = (now - lastAlertTs < ALERT_COOLDOWN_MS);

  if (!isCooldownActive || options.forceAlert) {
    lastAlertTime.set(alertKey, now);
    const alert = {
      alertId: crypto.randomUUID(),
      plate: normPlate,
      cameraId: detection.cameraStringId || `CAM_0${detection.cameraId}`,
      cameraNumericId: Number(detection.cameraId),
      cameraName: detection.cameraName || `CAM-${detection.cameraId}`,
      cameraLocation: detection.cameraLocation || 'Junction A',
      location: detection.cameraLocation || 'Junction A',
      timestamp: detection.timestamp || new Date(now).toISOString(),
      alertTime: new Date(now).toISOString(),
      reason: wlEntry.reason || 'Watchlisted vehicle',
      priority: wlEntry.priority || 'HIGH',
      notes: wlEntry.notes || '',
      ocrConfidence: detection.ocrConfidence !== undefined ? detection.ocrConfidence : 0.95,
      detectorConfidence: detection.detectorConfidence !== undefined ? detection.detectorConfidence : 0.96,
      confidence: detection.confidence !== undefined ? detection.confidence : 0.95,
      status: 'ACTIVE',
    };
    alertHistory.push(alert);
    return alert;
  }

  return { status: 'suppressed_by_cooldown', alertKey };
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Watchlist Record Creation & Validation
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 1. Watchlist Record Creation & Validation ---');

runTest('Adds vehicle to Watchlist without unnecessary PII', () => {
  const entry = addWatchlistEntry('TN45AB1234', 'Vehicle of interest - traffic violation', 'CRITICAL', 'Case #9021');
  assert.strictEqual(entry.plate, 'TN45AB1234');
  assert.strictEqual(entry.priority, 'CRITICAL');
  assert.strictEqual(entry.active, true);
  assert.strictEqual(entry.reason, 'Vehicle of interest - traffic violation');
  assert(entry.createdAt !== undefined);
  assert.strictEqual(entry.ownerName, undefined, 'No personal data should be stored');
  assert.strictEqual(entry.phone, undefined, 'No personal data should be stored');
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Alert Generation & Payload Schema Validation
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 2. Alert Generation & Payload Schema Validation ---');

let firstAlert = null;
runTest('Generates real-time alert with complete required metadata upon match', () => {
  const detection = {
    plate: 'TN45AB1234',
    cameraId: 1,
    cameraStringId: 'CAM_01',
    cameraName: 'Junction A',
    cameraLocation: 'Chennai Central',
    timestamp: '2026-09-28T10:00:00.000Z',
    ocrConfidence: 0.94,
    detectorConfidence: 0.96,
    confidence: 0.95,
  };

  firstAlert = processDetectionAlert(detection, { now: 1000000 });
  assert(firstAlert !== null, 'Alert must be generated');
  assert.strictEqual(firstAlert.plate, 'TN45AB1234');
  assert.strictEqual(firstAlert.cameraId, 'CAM_01');
  assert.strictEqual(firstAlert.cameraLocation, 'Chennai Central');
  assert.strictEqual(firstAlert.priority, 'CRITICAL');
  assert.strictEqual(firstAlert.status, 'ACTIVE');
  assert.strictEqual(firstAlert.ocrConfidence, 0.94);
  assert.strictEqual(firstAlert.detectorConfidence, 0.96);
  assert(firstAlert.alertId !== undefined && firstAlert.alertId.length > 10);
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Duplicate Alert Suppression (Cooldown Validation)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 3. Duplicate Alert Suppression (Cooldown Validation) ---');

runTest('Suppresses repeated detections of same plate at same camera within cooldown window', () => {
  const initialAlertCount = alertHistory.length;

  // Simulate 30 consecutive video frames over 10 seconds
  for (let f = 1; f <= 30; f++) {
    const det = {
      plate: 'TN45AB1234',
      cameraId: 1,
      cameraStringId: 'CAM_01',
      cameraLocation: 'Chennai Central',
      timestamp: new Date(1000000 + f * 333).toISOString(),
    };
    const res = processDetectionAlert(det, { now: 1000000 + f * 333 });
    assert.strictEqual(res.status, 'suppressed_by_cooldown');
  }

  // Alert count must remain exactly the same (no duplicate alerts created!)
  assert.strictEqual(alertHistory.length, initialAlertCount, 'No duplicate alerts should be recorded');
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cross-Camera Transition Alert Trigger
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 4. Cross-Camera Transition Alert Trigger ---');

runTest('Immediately triggers new alert when vehicle arrives at a different camera', () => {
  const detCam2 = {
    plate: 'TN45AB1234',
    cameraId: 2,
    cameraStringId: 'CAM_02',
    cameraName: 'Junction B',
    cameraLocation: 'Pantheon Road',
    timestamp: '2026-09-28T10:06:00.000Z',
  };

  // Even though it is within the 60s window of CAM_01, moving to CAM_02 is a meaningful event!
  const alertCam2 = processDetectionAlert(detCam2, { now: 1010000 });
  assert(alertCam2 !== null);
  assert.strictEqual(alertCam2.status, 'ACTIVE');
  assert.strictEqual(alertCam2.cameraId, 'CAM_02');
  assert.strictEqual(alertCam2.cameraLocation, 'Pantheon Road');
  assert.strictEqual(alertHistory.length, 2, 'Total alerts should now be 2 (one for CAM_01, one for CAM_02)');
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Cooldown Expiry Alert Retrigger
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 5. Cooldown Expiry Alert Retrigger ---');

runTest('Triggers new alert after cooldown period expires (> 60s)', () => {
  const detCam1Later = {
    plate: 'TN45AB1234',
    cameraId: 1,
    cameraStringId: 'CAM_01',
    cameraLocation: 'Chennai Central',
    timestamp: '2026-09-28T10:02:00.000Z',
  };

  // 120 seconds later at CAM_01 (1000000 + 120000)
  const alertLater = processDetectionAlert(detCam1Later, { now: 1120000 });
  assert(alertLater !== null);
  assert.strictEqual(alertLater.status, 'ACTIVE');
  assert.strictEqual(alertLater.cameraId, 'CAM_01');
  assert.strictEqual(alertHistory.length, 3);
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Inactive Watchlist Vehicle Suppression
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n--- 6. Inactive Watchlist Vehicle Suppression ---');

runTest('Does not generate alerts for deactivated watchlist entries', () => {
  const entry = watchlist.get('TN45AB1234');
  entry.active = false; // Deactivate

  const det = {
    plate: 'TN45AB1234',
    cameraId: 3,
    cameraStringId: 'CAM_03',
    timestamp: '2026-09-28T10:15:00.000Z',
  };

  const alert = processDetectionAlert(det, { now: 1200000 });
  assert.strictEqual(alert, null, 'No alert should be generated when watchlist record is inactive');
});

console.log('\n======================================================================');
console.log(`🏁 WATCHLIST TEST RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
console.log('======================================================================\n');

if (passedTests !== totalTests) {
  process.exit(1);
}
