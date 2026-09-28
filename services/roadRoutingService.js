'use strict';

/**
 * Road Routing & GIS Geometry Service
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 *
 * Implements OpenStreetMap-compatible road network routing (OSRM default),
 * camera road snapping, corridor leg stitching, multi-tier caching,
 * and graceful fallback handling.
 */

const fs = require('fs');
const path = require('path');
const { haversineDistance } = require('./geoService');

const CONFIG_FILE = path.join(__dirname, '..', 'data', 'analytics_config.json');
const TOPOLOGY_FILE = path.join(__dirname, '..', 'data', 'camera_network.json');
const CAMERAS_FILE = path.join(__dirname, '..', 'data', 'cameras.json');

// In-memory cache for fast lookup
const routeCache = new Map();
const snapCache = new Map();

/**
 * Loads analytics and routing configuration.
 */
function loadRoutingConfig() {
  let cfg = {};
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      cfg = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    }
  } catch (err) {
    console.warn('⚠️  Could not load analytics_config.json, using defaults:', err.message);
  }

  return {
    provider: process.env.ROUTING_PROVIDER || cfg.routingProvider || 'osrm',
    baseUrl: process.env.ROUTING_BASE_URL || cfg.routingBaseUrl || 'https://router.project-osrm.org',
    timeoutMs: Number(process.env.ROUTING_TIMEOUT_MS || cfg.routingTimeoutMs || 3500),
    maxSnapDistanceMeters: Number(cfg.maxSnapDistanceMeters || 250),
  };
}

/**
 * Loads pre-cached camera network topology from disk if present.
 */
function loadCameraTopology() {
  try {
    if (fs.existsSync(TOPOLOGY_FILE)) {
      return JSON.parse(fs.readFileSync(TOPOLOGY_FILE, 'utf8'));
    }
  } catch (err) {
    console.warn('⚠️  Could not load camera_network.json:', err.message);
  }
  return { cameras: [], routes: {} };
}

/**
 * Resolves camera object to standard format.
 */
function resolveCameraNode(camRef) {
  if (!camRef) return null;

  // If already standard object with lat/lng
  if (typeof camRef === 'object') {
    const lat = camRef.latitude !== undefined && camRef.latitude !== null ? Number(camRef.latitude) : (camRef.lat !== undefined && camRef.lat !== null ? Number(camRef.lat) : NaN);
    const lng = camRef.longitude !== undefined && camRef.longitude !== null ? Number(camRef.longitude) : (camRef.lng !== undefined && camRef.lng !== null ? Number(camRef.lng) : NaN);
    const id = camRef.cameraId !== undefined ? camRef.cameraId : (camRef.id !== undefined ? camRef.id : camRef.standardId);
    
    if (id === undefined && (isNaN(lat) || isNaN(lng))) {
      return null;
    }
    
    const numId = id !== undefined ? parseInt(String(id).replace(/[^0-9]/g, ''), 10) : null;
    const stdId = numId ? `CAM_${String(numId).padStart(2, '0')}` : 'CAM_CUSTOM';
    return {
      cameraId: numId || 0,
      standardId: stdId,
      name: camRef.name || camRef.cameraName || (numId ? `Camera ${numId}` : 'Custom Node'),
      roadName: camRef.roadName || camRef.location || '',
      latitude: lat,
      longitude: lng,
      lat,
      lng,
      direction: camRef.direction || camRef.allowedTravelDirection || 'unknown',
    };
  }

  // If camera identifier string or number (e.g., "CAM_01" or 1)
  const numId = parseInt(String(camRef).replace(/[^0-9]/g, ''), 10);
  const stdId = `CAM_${String(numId).padStart(2, '0')}`;

  // Check topology first
  const topology = loadCameraTopology();
  const foundInTopology = (topology.cameras || []).find(
    (c) => c.cameraId === numId || c.standardId === stdId
  );
  if (foundInTopology) {
    return {
      ...foundInTopology,
      lat: foundInTopology.latitude,
      lng: foundInTopology.longitude,
    };
  }

  // Check cameras.json
  try {
    if (fs.existsSync(CAMERAS_FILE)) {
      const cams = JSON.parse(fs.readFileSync(CAMERAS_FILE, 'utf8'));
      const found = cams.find((c) => c.cameraId === numId);
      if (found) {
        return {
          cameraId: found.cameraId,
          standardId: stdId,
          name: found.name,
          roadName: found.roadName,
          latitude: found.latitude,
          longitude: found.longitude,
          lat: found.latitude,
          lng: found.longitude,
          direction: found.direction || 'unknown',
        };
      }
    }
  } catch (e) {
    // ignore
  }

  return null;
}

/**
 * Snaps a camera coordinate to the nearest routable road segment.
 *
 * @param {object|string|number} camera
 * @param {object} [options]
 * @returns {Promise<object>}
 */
async function snapCameraToRoad(camera, options = {}) {
  const node = resolveCameraNode(camera);
  if (!node || isNaN(node.latitude) || isNaN(node.longitude)) {
    return {
      success: false,
      snapped: false,
      message: 'Invalid camera coordinates',
      originalCoordinate: null,
      snappedCoordinate: null,
      snapDistanceMeters: null,
    };
  }

  const cacheKey = `${node.standardId || node.cameraId}:${node.latitude.toFixed(6)},${node.longitude.toFixed(6)}`;
  if (snapCache.has(cacheKey) && !options.bypassCache) {
    return snapCache.get(cacheKey);
  }

  const cfg = loadRoutingConfig();
  const origCoord = [node.latitude, node.longitude];

  // Check topology pre-cached snap
  const topology = loadCameraTopology();
  const topoCam = (topology.cameras || []).find((c) => c.standardId === node.standardId);
  if (topoCam && topoCam.snappedCoordinate) {
    const snapResult = {
      success: true,
      snapped: true,
      cameraId: node.standardId,
      name: node.name,
      roadName: topoCam.nearestRoad || topoCam.roadName || node.roadName,
      originalCoordinate: origCoord,
      snappedCoordinate: topoCam.snappedCoordinate, // [lat, lng]
      snapDistanceMeters: topoCam.snapDistanceMeters || 0,
      provider: 'pre-cached-topology',
    };
    snapCache.set(cacheKey, snapResult);
    return snapResult;
  }

  // Query routing provider (OSRM nearest)
  if (cfg.provider === 'osrm' && typeof fetch === 'function') {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), cfg.timeoutMs);
      const url = `${cfg.baseUrl}/nearest/v1/driving/${node.longitude},${node.latitude}`;

      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        if (data.code === 'Ok' && data.waypoints && data.waypoints[0]) {
          const wp = data.waypoints[0];
          const dist = wp.distance || 0;
          if (dist <= cfg.maxSnapDistanceMeters) {
            const snapResult = {
              success: true,
              snapped: true,
              cameraId: node.standardId,
              name: node.name,
              roadName: wp.name || node.roadName,
              originalCoordinate: origCoord,
              snappedCoordinate: [wp.location[1], wp.location[0]], // [lat, lng]
              snapDistanceMeters: Number(dist.toFixed(2)),
              provider: 'osrm',
            };
            snapCache.set(cacheKey, snapResult);
            return snapResult;
          }
        }
      }
    } catch (err) {
      // Fallback to original coordinate
    }
  }

  // Graceful fallback to original coordinate
  const fallback = {
    success: true,
    snapped: false,
    cameraId: node.standardId,
    name: node.name,
    roadName: node.roadName,
    originalCoordinate: origCoord,
    snappedCoordinate: origCoord,
    snapDistanceMeters: 0,
    provider: 'identity-fallback',
  };
  snapCache.set(cacheKey, fallback);
  return fallback;
}

/**
 * Validates road network connectivity between two camera nodes.
 *
 * @param {object|string} cameraA
 * @param {object|string} cameraB
 * @returns {object}
 */
function validateRoadConnectivity(cameraA, cameraB) {
  const nodeA = resolveCameraNode(cameraA);
  const nodeB = resolveCameraNode(cameraB);

  if (!nodeA || !nodeB) {
    return {
      connected: false,
      reason: 'Invalid camera reference',
      nodeA: nodeA ? nodeA.standardId : null,
      nodeB: nodeB ? nodeB.standardId : null,
    };
  }

  if (nodeA.standardId === nodeB.standardId) {
    return {
      connected: false,
      reason: 'Identical camera node (stationary)',
      nodeA: nodeA.standardId,
      nodeB: nodeB.standardId,
    };
  }

  // Check topology routes
  const topology = loadCameraTopology();
  const routeKey = `${nodeA.standardId}:${nodeB.standardId}`;
  const preCachedRoute = topology.routes ? topology.routes[routeKey] : null;

  if (preCachedRoute) {
    return {
      connected: true,
      routeKey,
      fromCameraId: nodeA.standardId,
      toCameraId: nodeB.standardId,
      roadAligned: preCachedRoute.roadAligned,
      distanceKm: preCachedRoute.distanceKm,
      durationSeconds: preCachedRoute.durationSeconds,
      provider: preCachedRoute.provider || 'topology',
    };
  }

  // General geographic check
  const hDist = haversineDistance(nodeA.latitude, nodeA.longitude, nodeB.latitude, nodeB.longitude);
  return {
    connected: hDist !== null && hDist < 50.0, // within urban cluster
    routeKey,
    fromCameraId: nodeA.standardId,
    toCameraId: nodeB.standardId,
    roadAligned: false,
    distanceKm: hDist,
    provider: 'geodesic-fallback',
  };
}

/**
 * Synchronous road route lookup from in-memory cache or pre-cached camera topology.
 * Essential for fast synchronous API responses and backward compatibility.
 *
 * @param {object|string|number} start
 * @param {object|string|number} end
 * @returns {object}
 */
function getRoadRouteSync(start, end) {
  const nodeA = resolveCameraNode(start);
  const nodeB = resolveCameraNode(end);

  const latA = nodeA ? nodeA.latitude : (Array.isArray(start) ? start[0] : (start.lat || start.latitude));
  const lngA = nodeA ? nodeA.longitude : (Array.isArray(start) ? start[1] : (start.lng || start.longitude));
  const latB = nodeB ? nodeB.latitude : (Array.isArray(end) ? end[0] : (end.lat || end.latitude));
  const lngB = nodeB ? nodeB.longitude : (Array.isArray(end) ? end[1] : (end.lng || end.longitude));

  if (isNaN(latA) || isNaN(lngA) || isNaN(latB) || isNaN(lngB)) {
    return {
      success: false,
      roadAligned: false,
      routingStatus: 'UNAVAILABLE',
      message: 'Invalid start or end coordinates',
      geometry: null,
      latLngs: [],
      distanceMeters: 0,
      distanceKm: 0,
      durationSeconds: 0,
    };
  }

  if (Math.abs(latA - latB) < 1e-6 && Math.abs(lngA - lngB) < 1e-6) {
    return {
      success: true,
      roadAligned: true,
      routingStatus: 'OK',
      geometry: { type: 'LineString', coordinates: [[lngA, latA]] },
      latLngs: [[latA, lngA]],
      distanceMeters: 0,
      distanceKm: 0,
      durationSeconds: 0,
      provider: 'identity',
    };
  }

  const fromId = nodeA ? nodeA.standardId : `${latA.toFixed(4)},${lngA.toFixed(4)}`;
  const toId = nodeB ? nodeB.standardId : `${latB.toFixed(4)},${lngB.toFixed(4)}`;
  const cacheKey = `${fromId}:${toId}:driving`;

  if (routeCache.has(cacheKey)) {
    return routeCache.get(cacheKey);
  }

  // Check pre-cached camera topology
  const topology = loadCameraTopology();
  const topoKey = `${fromId}:${toId}`;
  if (topology.routes && topology.routes[topoKey]) {
    const cachedTopo = topology.routes[topoKey];
    const result = {
      success: true,
      fromCamera: fromId,
      toCamera: toId,
      roadAligned: true,
      routingStatus: 'OK',
      geometry: cachedTopo.geometry,
      latLngs: cachedTopo.latLngs || cachedTopo.geometry.coordinates.map((c) => [c[1], c[0]]),
      distanceMeters: cachedTopo.distanceMeters,
      distanceKm: cachedTopo.distanceKm,
      durationSeconds: cachedTopo.durationSeconds,
      provider: cachedTopo.provider || 'osrm-topology',
      cached: true,
    };
    routeCache.set(cacheKey, result);
    return result;
  }

  // Fallback if not found in topology cache
  const straightDistKm = haversineDistance(latA, lngA, latB, lngB);
  return {
    success: false,
    fromCamera: fromId,
    toCamera: toId,
    roadAligned: false,
    routingStatus: 'UNAVAILABLE',
    message: 'Road route unavailable in offline cache',
    geometry: {
      type: 'LineString',
      coordinates: [
        [lngA, latA],
        [lngB, latB],
      ],
    },
    latLngs: [
      [latA, lngA],
      [latB, lngB],
    ],
    distanceMeters: straightDistKm ? Math.round(straightDistKm * 1000) : 0,
    distanceKm: straightDistKm ? Number(straightDistKm.toFixed(2)) : 0,
    durationSeconds: straightDistKm ? Math.round((straightDistKm / 40.0) * 3600) : 0,
    provider: 'fallback-unavailable',
  };
}

/**
 * Synchronous multi-camera sequence route stitching from pre-cached topology.
 *
 * @param {Array<object|string|number>} cameraSequence
 * @returns {object}
 */
function getRoadRouteThroughCamerasSync(cameraSequence) {
  if (!Array.isArray(cameraSequence) || cameraSequence.length < 2) {
    const singleNode = cameraSequence && cameraSequence[0] ? resolveCameraNode(cameraSequence[0]) : null;
    return {
      success: true,
      roadAligned: true,
      routingStatus: 'OK',
      legs: [],
      latLngs: singleNode ? [[singleNode.latitude, singleNode.longitude]] : [],
      geometry: singleNode
        ? { type: 'LineString', coordinates: [[singleNode.longitude, singleNode.latitude]] }
        : null,
      totalDistanceMeters: 0,
      totalDistanceKm: 0,
      totalDurationSeconds: 0,
    };
  }

  const legs = [];
  const combinedLatLngs = [];
  const combinedCoordinates = [];
  let totalDistanceMeters = 0;
  let totalDurationSeconds = 0;
  let allRoadAligned = true;

  for (let i = 1; i < cameraSequence.length; i++) {
    const legRoute = getRoadRouteSync(cameraSequence[i - 1], cameraSequence[i]);
    legs.push(legRoute);

    if (!legRoute.roadAligned) {
      allRoadAligned = false;
    }

    totalDistanceMeters += legRoute.distanceMeters || 0;
    totalDurationSeconds += legRoute.durationSeconds || 0;

    const legPoints = legRoute.latLngs || [];
    const legCoords = (legRoute.geometry && legRoute.geometry.coordinates) || [];

    const startIdx = combinedLatLngs.length > 0 ? 1 : 0;
    for (let p = startIdx; p < legPoints.length; p++) {
      combinedLatLngs.push(legPoints[p]);
      if (legCoords[p]) {
        combinedCoordinates.push(legCoords[p]);
      }
    }
  }

  return {
    success: true,
    roadAligned: allRoadAligned,
    routingStatus: allRoadAligned ? 'OK' : 'PARTIALLY_UNAVAILABLE',
    legs,
    latLngs: combinedLatLngs,
    geometry: {
      type: 'LineString',
      coordinates: combinedCoordinates,
    },
    totalDistanceMeters,
    totalDistanceKm: Number((totalDistanceMeters / 1000).toFixed(2)),
    totalDurationSeconds,
    provider: 'osrm-multi-leg-sync',
  };
}

/**
 * Obtains road route geometry and metrics between two geographical endpoints or cameras.
 *
 * @param {object|Array<number>|string} start [lat, lng] or camera
 * @param {object|Array<number>|string} end [lat, lng] or camera
 * @param {object} [options]
 * @returns {Promise<object>}
 */
async function getRoadRoute(start, end, options = {}) {
  const nodeA = resolveCameraNode(start);
  const nodeB = resolveCameraNode(end);

  const latA = nodeA ? nodeA.latitude : (Array.isArray(start) ? start[0] : (start.lat || start.latitude));
  const lngA = nodeA ? nodeA.longitude : (Array.isArray(start) ? start[1] : (start.lng || start.longitude));
  const latB = nodeB ? nodeB.latitude : (Array.isArray(end) ? end[0] : (end.lat || end.latitude));
  const lngB = nodeB ? nodeB.longitude : (Array.isArray(end) ? end[1] : (end.lng || end.longitude));

  if (isNaN(latA) || isNaN(lngA) || isNaN(latB) || isNaN(lngB)) {
    return {
      success: false,
      roadAligned: false,
      routingStatus: 'UNAVAILABLE',
      message: 'Invalid start or end coordinates',
      geometry: null,
      latLngs: [],
      distanceMeters: 0,
      distanceKm: 0,
      durationSeconds: 0,
    };
  }

  // Same coordinate check
  if (Math.abs(latA - latB) < 1e-6 && Math.abs(lngA - lngB) < 1e-6) {
    return {
      success: true,
      roadAligned: true,
      routingStatus: 'OK',
      geometry: { type: 'LineString', coordinates: [[lngA, latA]] },
      latLngs: [[latA, lngA]],
      distanceMeters: 0,
      distanceKm: 0,
      durationSeconds: 0,
      provider: 'identity',
    };
  }

  const fromId = nodeA ? nodeA.standardId : `${latA.toFixed(4)},${lngA.toFixed(4)}`;
  const toId = nodeB ? nodeB.standardId : `${latB.toFixed(4)},${lngB.toFixed(4)}`;
  const cacheKey = `${fromId}:${toId}:driving`;

  if (options.simulateFailure) {
    const straightDistKm = haversineDistance(latA, lngA, latB, lngB);
    return {
      success: false,
      fromCamera: fromId,
      toCamera: toId,
      roadAligned: false,
      routingStatus: 'UNAVAILABLE',
      message: 'Road route unavailable',
      geometry: {
        type: 'LineString',
        coordinates: [
          [lngA, latA],
          [lngB, latB],
        ],
      },
      latLngs: [
        [latA, lngA],
        [latB, lngB],
      ],
      distanceMeters: straightDistKm ? Math.round(straightDistKm * 1000) : 0,
      distanceKm: straightDistKm ? Number(straightDistKm.toFixed(2)) : 0,
      durationSeconds: straightDistKm ? Math.round((straightDistKm / 40.0) * 3600) : 0,
      provider: 'fallback-simulated-failure',
    };
  }

  if (routeCache.has(cacheKey) && !options.bypassCache) {
    return routeCache.get(cacheKey);
  }

  // 1. Check pre-cached camera topology (instant, offline-resilient)
  const topology = loadCameraTopology();
  const topoKey = `${fromId}:${toId}`;
  if (topology.routes && topology.routes[topoKey] && !options.bypassCache) {
    const cachedTopo = topology.routes[topoKey];
    const result = {
      success: true,
      fromCamera: fromId,
      toCamera: toId,
      roadAligned: true,
      routingStatus: 'OK',
      geometry: cachedTopo.geometry,
      latLngs: cachedTopo.latLngs || cachedTopo.geometry.coordinates.map((c) => [c[1], c[0]]),
      distanceMeters: cachedTopo.distanceMeters,
      distanceKm: cachedTopo.distanceKm,
      durationSeconds: cachedTopo.durationSeconds,
      provider: cachedTopo.provider || 'osrm-topology',
      cached: true,
    };
    routeCache.set(cacheKey, result);
    return result;
  }

  // 2. Query Routing Provider (OSRM)
  const cfg = loadRoutingConfig();
  if (cfg.provider === 'osrm' && typeof fetch === 'function') {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), cfg.timeoutMs);
      const url = `${cfg.baseUrl}/route/v1/driving/${lngA},${latA};${lngB},${latB}?overview=full&geometries=geojson&alternatives=false`;

      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
          const route = data.routes[0];
          const distM = Math.round(route.distance);
          const durS = Math.round(route.duration);
          const coords = route.geometry.coordinates; // [[lng, lat], ...]
          const latLngs = coords.map((pt) => [pt[1], pt[0]]);

          const result = {
            success: true,
            fromCamera: fromId,
            toCamera: toId,
            roadAligned: true,
            routingStatus: 'OK',
            geometry: route.geometry,
            latLngs,
            distanceMeters: distM,
            distanceKm: Number((distM / 1000).toFixed(2)),
            durationSeconds: durS,
            provider: 'osrm',
            cached: false,
          };
          routeCache.set(cacheKey, result);
          return result;
        }
      }
    } catch (err) {
      // Request failed or timed out
    }
  }

  // 3. Fallback when road routing is unavailable
  // DO NOT pretend straight line is a road route! Set roadAligned: false and routingStatus: UNAVAILABLE
  const straightDistKm = haversineDistance(latA, lngA, latB, lngB);
  const fallback = {
    success: false,
    fromCamera: fromId,
    toCamera: toId,
    roadAligned: false,
    routingStatus: 'UNAVAILABLE',
    message: 'Road route unavailable',
    geometry: {
      type: 'LineString',
      coordinates: [
        [lngA, latA],
        [lngB, latB],
      ],
    },
    latLngs: [
      [latA, lngA],
      [latB, lngB],
    ],
    distanceMeters: straightDistKm ? Math.round(straightDistKm * 1000) : 0,
    distanceKm: straightDistKm ? Number(straightDistKm.toFixed(2)) : 0,
    durationSeconds: straightDistKm ? Math.round((straightDistKm / 40.0) * 3600) : 0,
    provider: 'fallback-unavailable',
  };

  return fallback;
}

/**
 * Obtains alternative road routes between start and end.
 */
async function getAlternativeRoadRoutes(start, end, options = {}) {
  const primary = await getRoadRoute(start, end, options);
  return [primary];
}

/**
 * Reconstructs a complete road-aligned route across a multi-camera sequence.
 * Stitches leg-by-leg road geometry (e.g. CAM_01 -> CAM_02 -> CAM_03).
 *
 * @param {Array<object|string|number>} cameraSequence
 * @param {object} [options]
 * @returns {Promise<object>}
 */
async function getRoadRouteThroughCameras(cameraSequence, options = {}) {
  if (!Array.isArray(cameraSequence) || cameraSequence.length < 2) {
    const singleNode = cameraSequence && cameraSequence[0] ? resolveCameraNode(cameraSequence[0]) : null;
    return {
      success: true,
      roadAligned: true,
      routingStatus: 'OK',
      legs: [],
      latLngs: singleNode ? [[singleNode.latitude, singleNode.longitude]] : [],
      geometry: singleNode
        ? { type: 'LineString', coordinates: [[singleNode.longitude, singleNode.latitude]] }
        : null,
      totalDistanceMeters: 0,
      totalDistanceKm: 0,
      totalDurationSeconds: 0,
    };
  }

  const legs = [];
  const combinedLatLngs = [];
  const combinedCoordinates = [];
  let totalDistanceMeters = 0;
  let totalDurationSeconds = 0;
  let allRoadAligned = true;

  for (let i = 1; i < cameraSequence.length; i++) {
    const prevCam = cameraSequence[i - 1];
    const currCam = cameraSequence[i];

    const legRoute = await getRoadRoute(prevCam, currCam, options);
    legs.push(legRoute);

    if (!legRoute.roadAligned) {
      allRoadAligned = false;
    }

    totalDistanceMeters += legRoute.distanceMeters || 0;
    totalDurationSeconds += legRoute.durationSeconds || 0;

    const legPoints = legRoute.latLngs || [];
    const legCoords = (legRoute.geometry && legRoute.geometry.coordinates) || [];

    // Avoid duplicate point at joint
    const startIdx = combinedLatLngs.length > 0 ? 1 : 0;
    for (let p = startIdx; p < legPoints.length; p++) {
      combinedLatLngs.push(legPoints[p]);
      if (legCoords[p]) {
        combinedCoordinates.push(legCoords[p]);
      }
    }
  }

  return {
    success: true,
    roadAligned: allRoadAligned,
    routingStatus: allRoadAligned ? 'OK' : 'PARTIALLY_UNAVAILABLE',
    legs,
    latLngs: combinedLatLngs,
    geometry: {
      type: 'LineString',
      coordinates: combinedCoordinates,
    },
    totalDistanceMeters,
    totalDistanceKm: Number((totalDistanceMeters / 1000).toFixed(2)),
    totalDurationSeconds,
    provider: 'osrm-multi-leg',
  };
}

/**
 * Helper accessors
 */
function getRouteDistance(route) {
  if (!route) return 0;
  return route.totalDistanceKm !== undefined
    ? route.totalDistanceKm
    : (route.distanceKm !== undefined ? route.distanceKm : 0);
}

function getRouteDuration(route) {
  if (!route) return 0;
  return route.totalDurationSeconds !== undefined
    ? route.totalDurationSeconds
    : (route.durationSeconds !== undefined ? route.durationSeconds : 0);
}

function clearRoutingCache() {
  routeCache.clear();
  snapCache.clear();
}

module.exports = {
  loadRoutingConfig,
  loadCameraTopology,
  resolveCameraNode,
  snapCameraToRoad,
  validateRoadConnectivity,
  getRoadRoute,
  getRoadRouteSync,
  getAlternativeRoadRoutes,
  getRoadRouteThroughCameras,
  getRoadRouteThroughCamerasSync,
  getRouteDistance,
  getRouteDuration,
  clearRoutingCache,
};
