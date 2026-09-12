'use strict';

const express    = require('express');
const https      = require('https');
const { Server } = require('socket.io');
const selfsigned = require('selfsigned');
const QRCode     = require('qrcode');
const path       = require('path');
const os         = require('os');
const fs         = require('fs');
const crypto     = require('crypto');

// ─────────────────────────────────────────────
//  Utilities
// ─────────────────────────────────────────────
function getLocalIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) return iface.address;
    }
  }
  return '127.0.0.1';
}

// ─────────────────────────────────────────────
//  TLS Certificate
// ─────────────────────────────────────────────
console.log('🔑  Generating self-signed TLS certificate...');
const attrs = [{ name: 'commonName', value: 'camera-monitor.local' }];
const pems  = selfsigned.generate(attrs, { days: 365, algorithm: 'sha256', keySize: 2048 });

// ─────────────────────────────────────────────
//  Express + HTTPS Server
// ─────────────────────────────────────────────
const app      = express();
const PORT     = 3000;
const HOST_IP  = getLocalIP();

const httpsServer = https.createServer({ key: pems.private, cert: pems.cert }, app);
const io = new Server(httpsServer, {
  cors: { origin: '*' },
  pingTimeout: 10000,
  pingInterval: 5000,
});

app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ─────────────────────────────────────────────
//  Camera Node Configuration
// ─────────────────────────────────────────────
const CAMERA_NODES = [
  { id: 1, name: 'Junction A',    lat: 13.0827, lng: 80.2707, location: 'Chennai Central',  zone: 'North' },
  { id: 2, name: 'Junction B',    lat: 13.0731, lng: 80.2609, location: 'T. Nagar',         zone: 'West'  },
  { id: 3, name: 'Junction C',    lat: 13.0878, lng: 80.2785, location: 'Perambur',         zone: 'North-East' },
  { id: 4, name: 'Highway Entry', lat: 13.0569, lng: 80.2425, location: 'Guindy',           zone: 'South' },
];

// ─────────────────────────────────────────────
//  Data Persistence
// ─────────────────────────────────────────────
const DATA_DIR   = path.join(__dirname, 'data');
const DET_FILE   = path.join(DATA_DIR, 'detections.json');
const WL_FILE    = path.join(DATA_DIR, 'watchlist.json');
const ALERT_FILE = path.join(DATA_DIR, 'alerts.json');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const MAX_DETECTIONS    = 10000;
const DETECTION_COOLDOWN = 30000; // ms — same plate+camera cooldown

let detections   = [];
let alertHistory = [];
let watchlist    = new Map(); // plate → { reason, addedAt, plate }
const lastDetTime = new Map(); // `${plate}:${cameraId}` → timestamp ms

function loadData() {
  try {
    if (fs.existsSync(DET_FILE))   detections   = JSON.parse(fs.readFileSync(DET_FILE, 'utf8'));
    if (fs.existsSync(ALERT_FILE)) alertHistory = JSON.parse(fs.readFileSync(ALERT_FILE, 'utf8'));
    if (fs.existsSync(WL_FILE)) {
      const arr = JSON.parse(fs.readFileSync(WL_FILE, 'utf8'));
      watchlist = new Map(arr);
    }
    console.log(`📦  Loaded ${detections.length} detections, ${watchlist.size} watchlist entries`);
  } catch (e) {
    console.warn('⚠️  Could not load saved data:', e.message);
  }
}

function saveData() {
  try {
    fs.writeFileSync(DET_FILE,   JSON.stringify(detections.slice(-MAX_DETECTIONS)));
    fs.writeFileSync(ALERT_FILE, JSON.stringify(alertHistory.slice(-2000)));
    fs.writeFileSync(WL_FILE,    JSON.stringify([...watchlist.entries()]));
  } catch (e) {
    console.warn('⚠️  Save failed:', e.message);
  }
}

loadData();
setInterval(saveData, 30000);
process.on('SIGINT',  () => { saveData(); process.exit(); });
process.on('SIGTERM', () => { saveData(); process.exit(); });

// ─────────────────────────────────────────────
//  Core Detection Store Function
// ─────────────────────────────────────────────
function storeDetection(data) {
  const { plate, cameraId, confidence, vehicleType, simulated } = data;
  if (!plate || !cameraId) return null;

  const camId = parseInt(cameraId, 10);
  const key   = `${plate.toUpperCase()}:${camId}`;
  const now   = Date.now();
  const last  = lastDetTime.get(key) || 0;

  if (now - last < DETECTION_COOLDOWN) return { status: 'cooldown' };
  lastDetTime.set(key, now);

  const camNode = CAMERA_NODES.find(c => c.id === camId) || {};
  const detection = {
    id:         crypto.randomUUID(),
    plate:      plate.toUpperCase().replace(/[^A-Z0-9]/g, ''),
    cameraId:   camId,
    cameraName: camNode.name     || `CAM-${camId}`,
    location:   camNode.location || '',
    zone:       camNode.zone     || '',
    timestamp:  new Date().toISOString(),
    confidence: Math.min(1, Math.max(0, parseFloat(confidence) || 0.85)),
    vehicleType: vehicleType || 'car',
    lat:        camNode.lat || 0,
    lng:        camNode.lng || 0,
    simulated:  !!simulated,
  };

  detections.push(detection);
  if (detections.length > MAX_DETECTIONS) detections.shift();

  // Broadcast live detection
  io.emit('anpr:detection', detection);

  // Watchlist check
  if (watchlist.has(detection.plate)) {
    const wlEntry = watchlist.get(detection.plate);
    const alert = {
      ...detection,
      alertId:   crypto.randomUUID(),
      reason:    wlEntry.reason,
      alertTime: new Date().toISOString(),
    };
    alertHistory.push(alert);
    io.emit('anpr:alert', alert);
    console.log(`🚨  WATCHLIST HIT: ${detection.plate} at ${detection.cameraName}`);
  }

  return detection;
}

// ─────────────────────────────────────────────
//  REST API — Original Endpoints (preserved)
// ─────────────────────────────────────────────
app.get('/api/network-info', (req, res) => {
  res.json({ ip: HOST_IP, port: PORT });
});

app.get('/api/qrcode/:id', async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (id < 1 || id > 3) return res.status(400).json({ error: 'Invalid camera id' });
  const url = `https://${HOST_IP}:${PORT}/camera.html?id=${id}`;
  try {
    const dataUrl = await QRCode.toDataURL(url, {
      errorCorrectionLevel: 'M', margin: 2,
      color: { dark: '#00ff41', light: '#0a0a0f' }, width: 280,
    });
    res.json({ qr: dataUrl, url });
  } catch (err) {
    res.status(500).json({ error: 'QR generation failed' });
  }
});

// ─────────────────────────────────────────────
//  REST API — ANPR / Intelligence Endpoints
// ─────────────────────────────────────────────

/** Camera node configuration */
app.get('/api/cameras/config', (req, res) => {
  const connectedIds = new Set(Object.keys(cameras).map(Number));
  const result = CAMERA_NODES.map(n => ({
    ...n,
    online: connectedIds.has(n.id) || n.id === 4,
    detectionCount: detections.filter(d => d.cameraId === n.id).length,
  }));
  res.json(result);
});

/** Store a detection (from browser ANPR engine via HTTP fallback) */
app.post('/api/detections', (req, res) => {
  const result = storeDetection(req.body);
  if (!result) return res.status(400).json({ error: 'Missing plate or cameraId' });
  if (result.status === 'cooldown') {
    const norm = req.body.plate ? req.body.plate.toUpperCase().replace(/[^A-Z0-9]/g, '') : '';
    const isWl = watchlist.has(norm);
    return res.json({
      status: 'cooldown',
      plate: norm,
      isWatchlisted: isWl,
      watchlistReason: isWl ? watchlist.get(norm).reason : null
    });
  }
  const isWl = watchlist.has(result.plate);
  res.json({
    status: 'stored',
    id: result.id,
    plate: result.plate,
    detection: result,
    isWatchlisted: isWl,
    watchlistReason: isWl ? watchlist.get(result.plate).reason : null
  });
});

/** Forward frame to Python ANPR Inference service */
app.post('/api/anpr/detect', async (req, res) => {
  try {
    const pyResp = await fetch('http://127.0.0.1:5001/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    const data = await pyResp.json();
    res.json(data);
  } catch (err) {
    res.status(503).json({ success: false, error: 'ANPR Inference service unavailable', detail: err.message });
  }
});

/** Query detections */
app.get('/api/detections', (req, res) => {
  let results = [...detections];
  const { plate, cameraId, limit, since } = req.query;

  if (plate)    results = results.filter(d => d.plate === plate.toUpperCase());
  if (cameraId) results = results.filter(d => d.cameraId === parseInt(cameraId));
  if (since)    results = results.filter(d => new Date(d.timestamp) >= new Date(since));

  results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  const lim = Math.min(parseInt(limit) || 100, 500);
  res.json(results.slice(0, lim));
});

/** Vehicle trajectory — all detections for a plate, chronological */
app.get('/api/detections/trajectory/:plate', (req, res) => {
  const plate = req.params.plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
  const trail = detections
    .filter(d => d.plate === plate)
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  if (trail.length === 0) return res.json({ plate, found: false, trail: [] });

  const first = trail[0];
  const last  = trail[trail.length - 1];
  const totalMs = new Date(last.timestamp) - new Date(first.timestamp);

  // Camera-to-camera segments
  const segments = [];
  for (let i = 1; i < trail.length; i++) {
    const diffMs = new Date(trail[i].timestamp) - new Date(trail[i-1].timestamp);
    segments.push({
      from:       trail[i-1].cameraName,
      to:         trail[i].cameraName,
      travelMins: Math.round(diffMs / 60000),
    });
  }

  res.json({
    plate,
    found:         true,
    trail,
    totalSightings: trail.length,
    firstSeen:     first.timestamp,
    lastSeen:      last.timestamp,
    totalMinutes:  Math.round(totalMs / 60000),
    cameras:       [...new Set(trail.map(d => d.cameraName))],
    segments,
  });
});

/** Traffic analytics aggregations */
app.get('/api/detections/analytics', (req, res) => {
  const now = Date.now();
  const last60min = detections.filter(d => now - new Date(d.timestamp).getTime() < 3600000);
  const last5min  = detections.filter(d => now - new Date(d.timestamp).getTime() < 300000);

  // Per camera counts
  const perCamera = {};
  CAMERA_NODES.forEach(n => { perCamera[n.id] = { name: n.name, count: 0 }; });
  last60min.forEach(d => { if (perCamera[d.cameraId]) perCamera[d.cameraId].count++; });

  // Busiest camera
  const busiestCam = Object.values(perCamera).sort((a, b) => b.count - a.count)[0];

  // Vehicles/min (last 5min)
  const vpm = last5min.length / 5;

  // Common routes (camera-to-camera pairs)
  const routeCounts = {};
  const plateGroups = {};
  last60min.forEach(d => {
    if (!plateGroups[d.plate]) plateGroups[d.plate] = [];
    plateGroups[d.plate].push(d);
  });
  Object.values(plateGroups).forEach(trail => {
    const sorted = trail.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    for (let i = 1; i < sorted.length; i++) {
      const key = `${sorted[i-1].cameraName} → ${sorted[i].cameraName}`;
      routeCounts[key] = (routeCounts[key] || 0) + 1;
    }
  });
  const topRoutes = Object.entries(routeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([route, count]) => ({ route, count }));

  // Hourly timeline (last 24 hours, per hour)
  const hourlyMap = {};
  detections.forEach(d => {
    const h = new Date(d.timestamp).toISOString().slice(0, 13);
    hourlyMap[h] = (hourlyMap[h] || 0) + 1;
  });
  const hourlyTimeline = Object.entries(hourlyMap)
    .sort()
    .slice(-24)
    .map(([hour, count]) => ({ hour: hour.slice(11) + ':00', count }));

  // Peak hour
  const peakEntry = Object.entries(hourlyMap).sort((a, b) => b[1] - a[1])[0];

  res.json({
    totalDetections:  detections.length,
    last60min:        last60min.length,
    vehiclesPerMin:   Math.round(vpm * 10) / 10,
    busiestCamera:    busiestCam?.name || 'N/A',
    perCamera:        Object.values(perCamera),
    topRoutes,
    hourlyTimeline,
    peakHour:         peakEntry ? peakEntry[0].slice(11) + ':00' : 'N/A',
    peakHourCount:    peakEntry ? peakEntry[1] : 0,
    uniquePlates:     new Set(detections.map(d => d.plate)).size,
    simulatedCount:   detections.filter(d => d.simulated).length,
  });
});

/** Watchlist */
app.get('/api/watchlist', (req, res) => {
  const entries = [...watchlist.entries()].map(([plate, data]) => ({
    plate,
    ...data,
    alertCount: alertHistory.filter(a => a.plate === plate).length,
  }));
  res.json(entries);
});

app.post('/api/watchlist', (req, res) => {
  const { plate, reason } = req.body;
  if (!plate) return res.status(400).json({ error: 'plate required' });
  const normalized = plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
  watchlist.set(normalized, { reason: reason || 'Suspicious vehicle', addedAt: new Date().toISOString(), plate: normalized });
  io.emit('watchlist:updated', [...watchlist.entries()].map(([p, d]) => ({ plate: p, ...d })));
  saveData();
  res.json({ status: 'added', plate: normalized });
});

app.delete('/api/watchlist/:plate', (req, res) => {
  const plate = req.params.plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
  watchlist.delete(plate);
  io.emit('watchlist:updated', [...watchlist.entries()].map(([p, d]) => ({ plate: p, ...d })));
  saveData();
  res.json({ status: 'removed', plate });
});

app.get('/api/alerts', (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 200);
  res.json(alertHistory.slice(-limit).reverse());
});

// ─────────────────────────────────────────────
//  Existing WebRTC State Tracking (preserved)
// ─────────────────────────────────────────────
const cameras         = {};
const dashboardSockets = new Set();

function broadcastCameraStatus() {
  const status = {};
  for (let i = 1; i <= 3; i++) { status[i] = cameras[i] ? 'connected' : 'disconnected'; }
  io.emit('cameras:status', status);
}

// ─────────────────────────────────────────────
//  Socket.io — Signaling (preserved) + ANPR events
// ─────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`🔌  Socket connected: ${socket.id}`);

  // ── Dashboard registers ──
  socket.on('dashboard:register', () => {
    dashboardSockets.add(socket.id);
    console.log(`🖥   Dashboard registered: ${socket.id}`);
    broadcastCameraStatus();
    // Send initial watchlist to new dashboard
    socket.emit('watchlist:updated', [...watchlist.entries()].map(([p, d]) => ({ plate: p, ...d })));
  });

  // ── Phone camera registers ──
  socket.on('camera:register', ({ cameraId }) => {
    const id = parseInt(cameraId, 10);
    if (id < 1 || id > 3) return;
    if (cameras[id] && cameras[id].socketId !== socket.id) {
      console.log(`⚠️   Replacing camera ${id} (old: ${cameras[id].socketId})`);
    }
    cameras[id] = { socketId: socket.id, status: 'connected' };
    socket.data.cameraId = id;
    console.log(`📷  Camera ${id} registered: ${socket.id}`);
    dashboardSockets.forEach(dashId => io.to(dashId).emit('camera:ready', { cameraId: id }));
    broadcastCameraStatus();
  });

  // ── WebRTC Signaling relay (all preserved) ──
  socket.on('webrtc:offer', ({ cameraId, sdp }) => {
    console.log(`📡  Offer from camera ${cameraId}`);
    dashboardSockets.forEach(dashId => io.to(dashId).emit('webrtc:offer', { cameraId, sdp }));
  });

  socket.on('webrtc:answer', ({ cameraId, sdp }) => {
    console.log(`📡  Answer for camera ${cameraId}`);
    const cam = cameras[parseInt(cameraId, 10)];
    if (cam) io.to(cam.socketId).emit('webrtc:answer', { sdp });
  });

  socket.on('webrtc:ice:phone', ({ cameraId, candidate }) => {
    dashboardSockets.forEach(dashId => io.to(dashId).emit('webrtc:ice:phone', { cameraId, candidate }));
  });

  socket.on('webrtc:ice:dashboard', ({ cameraId, candidate }) => {
    const cam = cameras[parseInt(cameraId, 10)];
    if (cam) io.to(cam.socketId).emit('webrtc:ice:dashboard', { candidate });
  });

  socket.on('camera:restart', ({ cameraId }) => {
    console.log(`🔄  Camera ${cameraId} requesting restart`);
    dashboardSockets.forEach(dashId => io.to(dashId).emit('camera:restart', { cameraId }));
  });

  // ── NEW: ANPR Detection submitted by browser ANPR engine ──
  socket.on('anpr:submit', (data) => {
    storeDetection(data);
  });

  // ── Disconnect ──
  socket.on('disconnect', (reason) => {
    console.log(`❌  Socket disconnected: ${socket.id} (${reason})`);
    if (dashboardSockets.has(socket.id)) {
      dashboardSockets.delete(socket.id);
      console.log(`🖥   Dashboard unregistered: ${socket.id}`);
    }
    const camId = socket.data.cameraId;
    if (camId && cameras[camId] && cameras[camId].socketId === socket.id) {
      delete cameras[camId];
      console.log(`📷  Camera ${camId} disconnected`);
      dashboardSockets.forEach(dashId => io.to(dashId).emit('camera:disconnected', { cameraId: camId }));
      broadcastCameraStatus();
    }
  });
});

// ─────────────────────────────────────────────
//  Start Server
// ─────────────────────────────────────────────
httpsServer.listen(PORT, '0.0.0.0', () => {
  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║   🏙️   ANPR Traffic Intelligence Platform  — RUNNING          ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log(`║  Command Center →  https://${HOST_IP}:${PORT}                  `);
  console.log(`║  Vehicle Tracking→  https://${HOST_IP}:${PORT}/tracking.html   `);
  console.log(`║  Analytics       →  https://${HOST_IP}:${PORT}/analytics.html  `);
  console.log(`║  Camera 1        →  https://${HOST_IP}:${PORT}/camera.html?id=1`);
  console.log(`║  Camera 2        →  https://${HOST_IP}:${PORT}/camera.html?id=2`);
  console.log(`║  Camera 3        →  https://${HOST_IP}:${PORT}/camera.html?id=3`);
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log('║  ⚠️  Phones: accept the certificate warning once              ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
});
