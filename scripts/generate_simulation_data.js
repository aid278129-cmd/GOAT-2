'use strict';

const fs = require('fs');
const path = require('path');

const simDir = path.join(__dirname, '..', 'data', 'simulation');
if (!fs.existsSync(simDir)) fs.mkdirSync(simDir, { recursive: true });

// 12 Virtual Camera Nodes in Greater Chennai Urban Corridor
const cameras = [
  { cameraId: 1, standardId: 'CAM_01', name: 'Chennai Central Gate', latitude: 13.0827, longitude: 80.2707, roadName: 'Poonamallee High Rd', zone: 'North', allowedTravelDirection: 'westbound' },
  { cameraId: 2, standardId: 'CAM_02', name: 'Egmore Junction', latitude: 13.0784, longitude: 80.2605, roadName: 'Gandhi Irwin Rd', zone: 'Central', allowedTravelDirection: 'southbound' },
  { cameraId: 3, standardId: 'CAM_03', name: 'T. Nagar Panagal Park', latitude: 13.0418, longitude: 80.2341, roadName: 'Usman Rd', zone: 'West', allowedTravelDirection: 'southbound' },
  { cameraId: 4, standardId: 'CAM_04', name: 'Guindy Kathipara', latitude: 13.0067, longitude: 80.2030, roadName: 'GST Road', zone: 'South', allowedTravelDirection: 'southbound' },
  { cameraId: 5, standardId: 'CAM_05', name: 'Royapettah Clock Tower', latitude: 13.0583, longitude: 80.2642, roadName: 'Royapettah High Rd', zone: 'East', allowedTravelDirection: 'southbound' },
  { cameraId: 6, standardId: 'CAM_06', name: 'Nungambakkam High Rd', latitude: 13.0595, longitude: 80.2428, roadName: 'NH Road', zone: 'Central-West', allowedTravelDirection: 'westbound' },
  { cameraId: 7, standardId: 'CAM_07', name: 'Saidapet Bridge', latitude: 13.0210, longitude: 80.2229, roadName: 'Anna Salai', zone: 'South-Central', allowedTravelDirection: 'southbound' },
  { cameraId: 8, standardId: 'CAM_08', name: 'Chennai Airport Entry', latitude: 12.9815, longitude: 80.1637, roadName: 'Airport Apron Link', zone: 'South-Airport', allowedTravelDirection: 'southbound' },
  { cameraId: 9, standardId: 'CAM_09', name: 'Velachery Bypass', latitude: 12.9759, longitude: 80.2212, roadName: 'Velachery Main Rd', zone: 'South-East', allowedTravelDirection: 'eastbound' },
  { cameraId: 10, standardId: 'CAM_10', name: 'OMR IT Corridor Tidel', latitude: 12.9892, longitude: 80.2483, roadName: 'Rajiv Gandhi Salai', zone: 'IT-Corridor', allowedTravelDirection: 'southbound' },
  { cameraId: 11, standardId: 'CAM_11', name: 'Adyar Signal', latitude: 13.0064, longitude: 80.2575, roadName: 'Sardar Patel Rd', zone: 'South-East', allowedTravelDirection: 'eastbound' },
  { cameraId: 12, standardId: 'CAM_12', name: 'Mylapore Luz Corner', latitude: 13.0368, longitude: 80.2676, roadName: 'Kutchery Rd', zone: 'East', allowedTravelDirection: 'northbound' },
];

fs.writeFileSync(path.join(simDir, 'cameras.json'), JSON.stringify(cameras, null, 2));

// Adjacency graph with realistic branching
const graph = {
  CAM_01: ['CAM_02', 'CAM_05'],
  CAM_02: ['CAM_03', 'CAM_06'],
  CAM_03: ['CAM_04', 'CAM_07'],
  CAM_04: ['CAM_08', 'CAM_09'],
  CAM_05: ['CAM_06', 'CAM_12'],
  CAM_06: ['CAM_03', 'CAM_07'],
  CAM_07: ['CAM_04', 'CAM_11'],
  CAM_08: ['CAM_04'],
  CAM_09: ['CAM_10', 'CAM_11'],
  CAM_10: ['CAM_11', 'CAM_09'],
  CAM_11: ['CAM_12', 'CAM_07'],
  CAM_12: ['CAM_05', 'CAM_01'],
};

// Generate realistic simulated journeys
const detections = [];
const plates = [
  'TN01AB1111', 'TN02BC2222', 'TN03CD3333', 'TN04DE4444', 'TN05EF5555',
  'TN07FG6666', 'TN09GH7777', 'TN10HJ8888', 'TN22JK9999', 'TN14KL1010',
  'KA01AB2020', 'KA04CD3030', 'AP09EF4040', 'DL01GH5050', 'MH02JK6060',
  'TN07AB9988', 'TN09CD7766', 'TN10EF5544', 'TN22GH3322', 'TN14JK1100'
];

const baseTime = new Date('2026-09-20T06:00:00.000Z').getTime();
let detId = 1;

// Corridors with probabilistic branching
const corridorWeights = {
  // Morning peak favors IT Corridor / OMR
  morning: {
    'CAM_01->CAM_02': 0.7, 'CAM_01->CAM_05': 0.3,
    'CAM_02->CAM_03': 0.6, 'CAM_02->CAM_06': 0.4,
    'CAM_03->CAM_07': 0.65, 'CAM_03->CAM_04': 0.35,
    'CAM_07->CAM_11': 0.7, 'CAM_07->CAM_04': 0.3,
    'CAM_11->CAM_10': 0.8, 'CAM_11->CAM_12': 0.2,
  },
  // Evening peak favors returning North
  evening: {
    'CAM_04->CAM_08': 0.3, 'CAM_04->CAM_09': 0.7,
    'CAM_09->CAM_10': 0.4, 'CAM_09->CAM_11': 0.6,
    'CAM_11->CAM_12': 0.7, 'CAM_11->CAM_07': 0.3,
    'CAM_12->CAM_05': 0.8, 'CAM_12->CAM_01': 0.2,
    'CAM_05->CAM_01': 0.7, 'CAM_05->CAM_06': 0.3,
  }
};

// Generate 180 multi-camera trips spanning 7 days
for (let tripIdx = 0; tripIdx < 180; tripIdx++) {
  const plate = plates[tripIdx % plates.length];
  const hour = (tripIdx * 3) % 24;
  const dayOffset = Math.floor(tripIdx / 25) * 86400000;
  let tripTime = baseTime + dayOffset + (hour * 3600000) + ((tripIdx % 15) * 60000);

  // Pick start camera
  let currCam = ['CAM_01', 'CAM_05', 'CAM_12', 'CAM_08'][tripIdx % 4];
  const tripLength = 3 + (tripIdx % 4); // 3 to 6 hops

  for (let hop = 0; hop < tripLength; hop++) {
    const camObj = cameras.find((c) => c.standardId === currCam);
    detections.push({
      id: 'sim-det-' + String(detId++).padStart(4, '0'),
      plate,
      cameraId: camObj.cameraId,
      cameraStringId: camObj.standardId,
      cameraNumericId: camObj.cameraId,
      timestamp: new Date(tripTime).toISOString(),
      confidence: 0.94,
      detectorConfidence: 0.95,
      ocrConfidence: 0.93,
      vehicleType: 'car',
      direction: camObj.allowedTravelDirection,
      latitude: camObj.latitude,
      longitude: camObj.longitude,
      lat: camObj.latitude,
      lng: camObj.longitude,
      cameraLocation: camObj.name,
      cameraName: camObj.name,
      zone: camObj.zone,
      simulated: true,
    });

    const nextOptions = graph[currCam] || [];
    if (nextOptions.length === 0) break;

    // Pick next camera with weighted probability depending on time-of-day
    let nextCam = nextOptions[0];
    if (nextOptions.length > 1) {
      const isMorning = hour >= 7 && hour <= 11;
      const weights = isMorning ? corridorWeights.morning : corridorWeights.evening;
      const key = `${currCam}->${nextOptions[0]}`;
      const p1 = weights[key] !== undefined ? weights[key] : 0.5;
      nextCam = ((tripIdx + hop) % 100) / 100 < p1 ? nextOptions[0] : nextOptions[1];
    }

    // Realistic urban travel time: 4 to 12 minutes (240s to 720s)
    const travelSec = 240 + ((tripIdx * 17 + hop * 31) % 480);
    tripTime += travelSec * 1000;
    currCam = nextCam;
  }
}

fs.writeFileSync(path.join(simDir, 'detections.json'), JSON.stringify(detections, null, 2));

// Camera network topology
const routes = {};
cameras.forEach((c1) => {
  cameras.forEach((c2) => {
    if (c1.cameraId !== c2.cameraId) {
      const key = `${c1.standardId}:${c2.standardId}`;
      const dLat = ((c2.latitude - c1.latitude) * Math.PI) / 180;
      const dLon = ((c2.longitude - c1.longitude) * Math.PI) / 180;
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos((c1.latitude * Math.PI) / 180) *
          Math.cos((c2.latitude * Math.PI) / 180) *
          Math.sin(dLon / 2) *
          Math.sin(dLon / 2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      const distKm = Number((6371 * c * 1.32).toFixed(2));
      const distMeters = Math.round(distKm * 1000);
      const durationSec = Math.round((distKm / 35) * 3600); // ~35 km/h urban speed
      routes[key] = {
        fromCamera: c1.standardId,
        toCamera: c2.standardId,
        roadAligned: true,
        distanceMeters: distMeters,
        distanceKm: distKm,
        durationSeconds: durationSec,
        provider: 'simulated-osm-topology',
      };
    }
  });
});

const simTopology = {
  version: '1.0.0-simulation',
  simulated: true,
  cameras: cameras.map((c) => ({
    ...c,
    nearestRoad: c.roadName,
    snappedCoordinate: [c.latitude, c.longitude],
    snapDistanceMeters: 4.2,
    possibleOutgoingRoadDirections: graph[c.standardId] ? ['southbound', 'eastbound', 'westbound'] : ['northbound'],
    connectedCameraIds: graph[c.standardId] || [],
  })),
  routes,
};

fs.writeFileSync(path.join(simDir, 'camera_network.json'), JSON.stringify(simTopology, null, 2));

console.log('✅ Simulation environment successfully created in data/simulation:');
console.log('  Cameras:', cameras.length);
console.log('  Detections:', detections.length);
console.log('  Routes:', Object.keys(routes).length);
