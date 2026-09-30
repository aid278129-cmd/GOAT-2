'use strict';

const http = require('http');

function postJSON(path, payload) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(payload);
    const req = http.request(
      'http://localhost:3001' + path,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            resolve(body);
          }
        });
      }
    );
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function getJSON(path) {
  return new Promise((resolve, reject) => {
    http.get('http://localhost:3001' + path, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          resolve(body);
        }
      });
    }).on('error', reject);
  });
}

(async () => {
  console.log('======================================================================');
  console.log('🧪 EXECUTING PHASE 30 END-TO-END LIVE PREDICTION TEST');
  console.log('======================================================================\n');

  const testPlate = 'E2ETEST' + Math.floor(1000 + Math.random() * 9000);
  const t0 = new Date().toISOString();

  // 1. CAM_01 detects vehicle
  console.log(`[Step 1] CAM_01 detects vehicle: ${testPlate}`);
  const det1 = await postJSON('/api/detections', {
    plate: testPlate,
    cameraId: 1,
    cameraName: 'Junction A',
    latitude: 13.0827,
    longitude: 80.2707,
    confidence: 0.95,
    timestamp: t0,
    vehicleType: 'car',
    direction: 'northbound',
  });
  console.log('  Detection 1 accepted:', det1.plate, 'at CAM_01');

  // 2. Query prediction
  console.log('\n[Step 2] Querying prediction for vehicle at CAM_01...');
  const pred1 = await getJSON(`/api/trajectory/${testPlate}/predict`);
  console.log('  Current Camera:', pred1.currentCamera);
  console.log('  Top Candidates:');
  pred1.nextCameras.forEach((c, idx) => {
    console.log(
      `   #${idx + 1}: ${c.cameraId} (${Math.round(c.probability * 100)}%) - Confidence: ${c.confidence || c.confidenceTier} - ETA: ${c.etaFormatted}`
    );
    if (c.evidence) {
      console.log(
        `       Evidence: ${c.evidence.firstOrderTransitions}/${c.evidence.totalOutgoingTransitions} transitions, Road: ${c.evidence.topologyVerified}, Dir: ${c.evidence.directionCompatible}`
      );
    }
  });

  const predictedTop = pred1.nextCameras[0];

  // 3. Vehicle arrives at next camera (CAM_02) after realistic travel time
  console.log(`\n[Step 3] Vehicle detected at next camera: CAM_02`);
  // Simulated arrival 180s later
  const t1 = new Date(Date.now() + 180000).toISOString();
  const det2 = await postJSON('/api/detections', {
    plate: testPlate,
    cameraId: 2,
    cameraName: 'Junction B',
    latitude: 13.0731,
    longitude: 80.2609,
    confidence: 0.94,
    timestamp: t1,
    vehicleType: 'car',
    direction: 'eastbound',
  });
  console.log('  Detection 2 accepted:', det2.plate, 'at CAM_02');

  // 4. Query Prediction Evaluation History
  console.log('\n[Step 4] Checking automatic prediction resolution in evaluation history...');
  const history = await getJSON('/api/trajectory/predictions/history');
  const resolved = history.history.find((h) => h.plate === testPlate);

  if (resolved) {
    console.log('  ✅ PREDICTION AUTOMATICALLY RESOLVED:');
    console.log('     Prediction ID:', resolved.predictionId);
    console.log('     Predicted Camera:', resolved.predictedTopCamera);
    console.log('     Actual Camera:', resolved.actualNextCamera);
    console.log('     Probability:', Math.round(resolved.predictedProbability * 100) + '%');
    console.log('     Correct:', resolved.predictionCorrect ? 'YES (PASS)' : 'NO');
    console.log('     Top-3 Correct:', resolved.top3Correct ? 'YES (PASS)' : 'NO');
    console.log('     Actual Travel Time (s):', resolved.actualTravelTimeSeconds);
  } else {
    console.log('  Pending predictions in memory:', history.activePending);
  }

  // 5. Query updated trajectory
  console.log('\n[Step 5] Querying updated trajectory & next prediction from CAM_02...');
  const traj2 = await getJSON(`/api/trajectory/${testPlate}`);
  console.log('  Active Journey Sightings:', traj2.totalSightings);
  console.log('  Road Distance:', traj2.roadDistanceKm, 'km (Road-Aligned:', traj2.roadAligned, ')');
  console.log('  New Predictions from CAM_02:');
  traj2.predictions.forEach((c, idx) => {
    console.log(
      `   #${idx + 1}: ${c.cameraId} (${Math.round(c.probability * 100)}%) - Confidence: ${c.confidence || c.confidenceTier} - ETA: ${c.etaFormatted}`
    );
  });

  console.log('\n======================================================================');
  console.log('🏁 PHASE 30 TEST EXECUTION COMPLETE');
  console.log('======================================================================');
})();
