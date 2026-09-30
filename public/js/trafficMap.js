/**
 * trafficMap.js — Phase D GIS Traffic Intelligence Map
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 *
 * Responsibilities:
 * - Dynamic Leaflet visualization: Density Heatmap, Route Density Corridors,
 *   Congestion Bottlenecks, Camera Status Markers, and Vehicle Trajectory Tracking.
 * - Time range filtering (15m, 30m, 1h, Today, Custom).
 * - Debounced auto-refreshing of active layers only.
 * - Dynamic adaptive map legend and camera popups.
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Pure Data Transformation & Utility Functions (UMD Exportable for Testing)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Normalizes camera detection density for Leaflet.heat layer.
 * Formula: intensity = clamp(0.15, 1.0, metricValue / maxMetricValue)
 * Supports explicitly selected metrics: 'vehicleCount' (Traffic Volume), 'congestion', 'uniqueVehicles', 'cameraLoad'.
 * Handles zero max, null coordinates, and missing values safely.
 *
 * @param {Array<object>} camerasData - Output of GET /api/analytics/density
 * @param {string} [metric='vehicleCount'] - Metric key: 'vehicleCount' | 'congestion' | 'uniqueVehicles' | 'cameraLoad'
 * @returns {Array<Array<number>>} Array of [lat, lng, intensity]
 */
function normalizeHeatmapData(camerasData, metric = 'vehicleCount') {
  if (!Array.isArray(camerasData) || camerasData.length === 0) {
    return [];
  }

  // Filter valid coordinates
  const validCameras = camerasData.filter((c) => {
    const lat = Number(c.latitude !== undefined ? c.latitude : c.lat);
    const lng = Number(c.longitude !== undefined ? c.longitude : c.lng);
    return !isNaN(lat) && !isNaN(lng) && lat !== 0 && lng !== 0 && Math.abs(lat) <= 90 && Math.abs(lng) <= 180;
  });

  if (validCameras.length === 0) return [];

  // Determine value extractor based on metric
  const getValue = (c) => {
    if (metric === 'congestion') {
      const level = c.densityLevel || 'LOW';
      if (level === 'SEVERE') return 1.0;
      if (level === 'HIGH') return 0.75;
      if (level === 'MEDIUM') return 0.5;
      return 0.25;
    }
    if (metric === 'cameraLoad' || metric === 'totalDetections') {
      return Number(c.totalDetections !== undefined ? c.totalDetections : (c.rawDetections || c.vehicleCount)) || 0;
    }
    if (metric === 'uniqueVehicles' || metric === 'uniquePlateCount') {
      return Number(c.uniquePlateCount || c.uniqueVehicles || c.vehicleCount) || 0;
    }
    return Number(c.vehicleCount) || 0;
  };

  if (metric === 'congestion') {
    return validCameras.map((c) => {
      const lat = Number(c.latitude !== undefined ? c.latitude : c.lat);
      const lng = Number(c.longitude !== undefined ? c.longitude : c.lng);
      return [lat, lng, getValue(c)];
    });
  }

  const maxVal = Math.max(...validCameras.map(getValue), 0);

  return validCameras.map((c) => {
    const lat = Number(c.latitude !== undefined ? c.latitude : c.lat);
    const lng = Number(c.longitude !== undefined ? c.longitude : c.lng);
    const val = getValue(c);

    let intensity = 0;
    if (maxVal > 0 && val > 0) {
      const ratio = val / maxVal;
      // Clamp between 0.15 (subtle floor for visibility) and 1.0
      intensity = Math.max(0.15, Math.min(1.0, Number(ratio.toFixed(3))));
    }

    return [lat, lng, intensity];
  });
}

/**
 * Normalizes route volume to a sensible polyline pixel width (e.g., 2px to 12px).
 *
 * @param {number} vehicleCount
 * @param {number} maxVolume
 * @param {number} [minWidth=2]
 * @param {number} [maxWidth=12]
 * @returns {number}
 */
function normalizeRouteWidth(vehicleCount, maxVolume, minWidth = 2, maxWidth = 12) {
  const count = Number(vehicleCount) || 0;
  const max = Number(maxVolume) || 0;

  if (max <= 0 || count <= 0) {
    return minWidth;
  }

  const ratio = Math.min(1.0, Math.max(0.0, count / max));
  return Math.round(minWidth + ratio * (maxWidth - minWidth));
}

/**
 * Maps congestion level to visual styling tokens.
 *
 * @param {string} level - 'NORMAL' | 'MODERATE' | 'HIGH' | 'SEVERE'
 * @returns {object}
 */
function mapCongestionStyle(level) {
  switch ((level || '').toUpperCase()) {
    case 'SEVERE':
      return {
        color: '#ff3b30',
        weight: 7,
        opacity: 0.95,
        zIndex: 1000,
        badgeClass: 'badge danger',
        label: 'SEVERE',
      };
    case 'HIGH':
      return {
        color: '#ff6b00',
        weight: 5.5,
        opacity: 0.9,
        zIndex: 800,
        badgeClass: 'badge danger',
        label: 'HIGH',
      };
    case 'MODERATE':
      return {
        color: '#ffb400',
        weight: 4,
        opacity: 0.85,
        zIndex: 600,
        badgeClass: 'badge amber',
        label: 'MODERATE',
      };
    case 'NORMAL':
    default:
      return {
        color: '#00ff41',
        weight: 3,
        opacity: 0.75,
        zIndex: 400,
        badgeClass: 'badge green',
        label: 'NORMAL',
      };
  }
}

/**
 * Maps camera density tier to color and badge styling.
 *
 * @param {string} level - 'LOW' | 'MEDIUM' | 'HIGH' | 'SEVERE'
 * @returns {object}
 */
function mapDensityStyle(level) {
  switch ((level || '').toUpperCase()) {
    case 'SEVERE':
      return {
        color: '#ff3b30',
        glow: 'rgba(255, 59, 48, 0.45)',
        badgeClass: 'badge danger',
        text: 'SEVERE',
      };
    case 'HIGH':
      return {
        color: '#ff6b00',
        glow: 'rgba(255, 107, 0, 0.45)',
        badgeClass: 'badge danger',
        text: 'HIGH',
      };
    case 'MEDIUM':
      return {
        color: '#ffb400',
        glow: 'rgba(255, 180, 0, 0.45)',
        badgeClass: 'badge amber',
        text: 'MEDIUM',
      };
    case 'LOW':
    default:
      return {
        color: '#00ff41',
        glow: 'rgba(0, 255, 65, 0.45)',
        badgeClass: 'badge green',
        text: 'LOW',
      };
  }
}

/**
 * Resolves [lat, lng] coordinates for both endpoints of a route.
 * Checks route object fields first, then falls back to camera lookup map.
 *
 * @param {object} route
 * @param {object|Array<object>} [cameraLookup]
 * @returns {{ fromCoord: [number, number], toCoord: [number, number] } | null}
 */
function resolveRouteEndpoints(route, cameraLookup = {}) {
  if (!route) return null;

  // Direct coordinates on route object (from Phase C additive metadata)
  let fromCoord = null;
  let toCoord = null;

  if (Array.isArray(route.fromCoordinates) && route.fromCoordinates.length === 2) {
    fromCoord = [Number(route.fromCoordinates[0]), Number(route.fromCoordinates[1])];
  } else if (route.fromLatitude && route.fromLongitude) {
    fromCoord = [Number(route.fromLatitude), Number(route.fromLongitude)];
  }

  if (Array.isArray(route.toCoordinates) && route.toCoordinates.length === 2) {
    toCoord = [Number(route.toCoordinates[0]), Number(route.toCoordinates[1])];
  } else if (route.toLatitude && route.toLongitude) {
    toCoord = [Number(route.toLatitude), Number(route.toLongitude)];
  }

  // Fallback: search cameraLookup map or list
  if (!fromCoord || !toCoord) {
    const lookupMap = new Map();
    const list = Array.isArray(cameraLookup) ? cameraLookup : Object.values(cameraLookup || {});

    list.forEach((c) => {
      const lat = Number(c.latitude !== undefined ? c.latitude : c.lat);
      const lng = Number(c.longitude !== undefined ? c.longitude : c.lng);
      if (!isNaN(lat) && !isNaN(lng)) {
        if (c.cameraId) lookupMap.set(String(c.cameraId), [lat, lng]);
        if (c.standardId) lookupMap.set(String(c.standardId), [lat, lng]);
        if (c.id !== undefined) lookupMap.set(String(c.id), [lat, lng]);
      }
    });

    if (!fromCoord && route.fromCameraId) {
      fromCoord = lookupMap.get(String(route.fromCameraId)) || null;
    }
    if (!toCoord && route.toCameraId) {
      toCoord = lookupMap.get(String(route.toCameraId)) || null;
    }
  }

  // Validate coordinates
  const isValid = (pt) =>
    Array.isArray(pt) &&
    pt.length === 2 &&
    !isNaN(pt[0]) &&
    !isNaN(pt[1]) &&
    pt[0] !== 0 &&
    pt[1] !== 0 &&
    Math.abs(pt[0]) <= 90 &&
    Math.abs(pt[1]) <= 180;

  if (isValid(fromCoord) && isValid(toCoord)) {
    return { fromCoord, toCoord };
  }

  return null;
}

/**
 * Calculates time range ISO strings from preset filter or custom values.
 *
 * @param {string} filterType - '15m' | '30m' | '1h' | 'today' | 'custom'
 * @param {string|Date} [customFrom]
 * @param {string|Date} [customTo]
 * @param {number} [now=Date.now()]
 * @returns {{ valid: boolean, from?: string, to?: string, error?: string }}
 */
function calculateTimeRangeQuery(filterType, customFrom, customTo, now = Date.now()) {
  const nowDate = new Date(now);

  switch (filterType) {
    case '15m':
      return {
        valid: true,
        from: new Date(now - 15 * 60 * 1000).toISOString(),
        to: nowDate.toISOString(),
      };
    case '30m':
      return {
        valid: true,
        from: new Date(now - 30 * 60 * 1000).toISOString(),
        to: nowDate.toISOString(),
      };
    case '1h':
      return {
        valid: true,
        from: new Date(now - 60 * 60 * 1000).toISOString(),
        to: nowDate.toISOString(),
      };
    case 'today': {
      const todayStart = new Date(nowDate);
      todayStart.setHours(0, 0, 0, 0);
      return {
        valid: true,
        from: todayStart.toISOString(),
        to: nowDate.toISOString(),
      };
    }
    case 'custom': {
      if (!customFrom || !customTo) {
        return { valid: false, error: 'Custom range requires both From and To dates' };
      }
      const fDate = new Date(customFrom);
      const tDate = new Date(customTo);

      if (isNaN(fDate.getTime()) || isNaN(tDate.getTime())) {
        return { valid: false, error: 'Malformed date string in custom range' };
      }
      if (fDate >= tDate) {
        return { valid: false, error: 'Start date must be earlier than End date' };
      }
      return {
        valid: true,
        from: fDate.toISOString(),
        to: tDate.toISOString(),
      };
    }
    default:
      return { valid: true, from: null, to: null };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. City Traffic Map Controller (Browser Frontend)
// ─────────────────────────────────────────────────────────────────────────────

const TrafficMap = (function () {
  const state = {
    map: null,
    containerId: 'analyticsMap',
    mode: 'heatmap', // 'heatmap' | 'routes' | 'congestion' | 'trajectory' | 'cameras'
    heatmapMetric: 'vehicleCount', // 'vehicleCount' | 'congestion' | 'uniqueVehicles' | 'cameraLoad'
    showCameras: true,
    timeFilter: '15m',
    customFrom: null,
    customTo: null,
    refreshIntervalMs: 10000,
    refreshTimer: null,
    isRefreshing: false,
    lastRefreshTime: null,
    currentTrackedPlate: '',
    cameraDataCache: [],
    // Layer Groups
    layers: {
      cameras: null,
      heatmap: null,
      routes: null,
      congestion: null,
      trajectory: null,
    },
    legendControl: null,
    initialized: false,
  };

  /**
   * Initializes Leaflet Map and Layer Groups.
   */
  function init(containerId = 'analyticsMap') {
    if (state.initialized && state.map) {
      state.map.invalidateSize();
      return;
    }

    const container = document.getElementById(containerId);
    if (!container) return;

    state.containerId = containerId;

    // Initialize Map with dark tiles
    state.map = L.map(containerId, {
      zoomControl: true,
      attributionControl: false,
    }).setView([13.07, 80.26], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
    }).addTo(state.map);

    const pane = container.querySelector('.leaflet-tile-pane');
    if (pane) {
      pane.style.filter = 'invert(1) hue-rotate(180deg) brightness(0.85) contrast(0.9)';
    }

    // Initialize Layer Groups
    state.layers.cameras = L.layerGroup().addTo(state.map);
    state.layers.routes = L.layerGroup();
    state.layers.congestion = L.layerGroup();
    state.layers.trajectory = L.layerGroup();
    state.layers.heatmap = null; // Instantiated dynamically by L.heatLayer

    // Attach dynamic legend
    createLegendControl();

    // Initial load
    setMode(state.mode, false);
    refreshActiveLayer();

    // Start auto-refresh interval
    startAutoRefresh();

    state.initialized = true;

    // Ensure Leaflet calculates dimensions correctly
    setTimeout(() => {
      if (state.map) state.map.invalidateSize();
    }, 200);
  }

  /**
   * Creates the dynamic Leaflet legend in bottom-right corner.
   */
  function createLegendControl() {
    if (state.legendControl || !state.map) return;

    const Legend = L.Control.extend({
      options: { position: 'bottomright' },
      onAdd: function () {
        const div = L.DomUtil.create('div', 'traffic-map-legend');
        div.id = 'trafficMapLegend';
        div.style.background = 'rgba(12, 13, 20, 0.88)';
        div.style.border = '1px solid var(--border)';
        div.style.borderRadius = '6px';
        div.style.padding = '8px 12px';
        div.style.color = '#dde2ec';
        div.style.fontFamily = "'Share Tech Mono', monospace";
        div.style.fontSize = '11px';
        div.style.boxShadow = '0 4px 14px rgba(0,0,0,0.5)';
        div.style.backdropFilter = 'blur(6px)';
        div.style.maxWidth = '210px';
        div.style.lineHeight = '1.4';
        return div;
      },
    });

    state.legendControl = new Legend();
    state.legendControl.addTo(state.map);
    updateLegendUI();
  }

  /**
   * Updates the Legend HTML according to the active mode.
   */
  function updateLegendUI() {
    const el = document.getElementById('trafficMapLegend');
    if (!el) return;

    let html = '';

    if (state.mode === 'heatmap') {
      html = `
        <div style="font-weight:700;color:#00ff41;margin-bottom:4px;border-bottom:1px solid #1a1d2e;padding-bottom:2px;">🔥 TRAFFIC DENSITY</div>
        <div style="height:8px;border-radius:4px;background:linear-gradient(to right, #00ff41, #00b4ff, #ffb400, #ff3b30);margin:6px 0;"></div>
        <div style="display:flex;justify-content:space-between;color:var(--text-dim);font-size:10px;">
          <span>Low</span><span>Med</span><span>High</span><span>Severe</span>
        </div>
      `;
    } else if (state.mode === 'routes') {
      html = `
        <div style="font-weight:700;color:#00b4ff;margin-bottom:4px;border-bottom:1px solid #1a1d2e;padding-bottom:2px;">🛣 ROUTE FLOW USAGE</div>
        <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:24px;height:2px;background:#00b4ff;"></div><span style="font-size:10px;">Low Volume (2px)</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:24px;height:5px;background:#00b4ff;"></div><span style="font-size:10px;">Medium (5-7px)</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:24px;height:10px;background:#00b4ff;"></div><span style="font-size:10px;">High Flow (10-12px)</span>
          </div>
        </div>
      `;
    } else if (state.mode === 'congestion') {
      html = `
        <div style="font-weight:700;color:#ffb400;margin-bottom:4px;border-bottom:1px solid #1a1d2e;padding-bottom:2px;">🚦 CONGESTION STATUS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px;font-size:10px;">
          <div><span style="color:#00ff41;">■</span> Normal</div>
          <div><span style="color:#ffb400;">■</span> Moderate</div>
          <div><span style="color:#ff6b00;">■</span> High</div>
          <div><span style="color:#ff3b30;">■</span> Severe</div>
        </div>
      `;
    } else if (state.mode === 'trajectory') {
      html = `
        <div style="font-weight:700;color:#00ff41;margin-bottom:4px;border-bottom:1px solid #1a1d2e;padding-bottom:2px;">🚗 VEHICLE TRAJECTORY</div>
        <div style="font-size:10px;color:var(--text-dim);margin-top:2px;">
          <div>${state.currentTrackedPlate ? `Tracking: <b style="color:#fff;">${state.currentTrackedPlate}</b>` : 'Enter plate to track'}</div>
          <div style="margin-top:4px;"><span style="color:#00ff41;">━ ━</span> Chronological Path</div>
        </div>
      `;
    } else if (state.mode === 'cameras') {
      html = `
        <div style="font-weight:700;color:#00ff41;margin-bottom:4px;border-bottom:1px solid #1a1d2e;padding-bottom:2px;">📷 CAMERA STATUS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px;font-size:10px;">
          <div><span style="color:#00ff41;">●</span> Low</div>
          <div><span style="color:#ffb400;">●</span> Med</div>
          <div><span style="color:#ff6b00;">●</span> High</div>
          <div><span style="color:#ff3b30;">●</span> Severe</div>
        </div>
      `;
    }

    if (state.showCameras && state.mode !== 'cameras') {
      html += `
        <div style="border-top:1px solid #1a1d2e;margin-top:6px;padding-top:4px;font-size:10px;color:var(--text-dim);">
          <span style="color:#00ff41;">●</span> Node markers visible
        </div>
      `;
    }

    el.innerHTML = html;
  }

  /**
   * Sets the active map mode (layer exclusivity).
   *
   * @param {string} mode - 'heatmap' | 'routes' | 'congestion' | 'trajectory' | 'cameras'
   * @param {boolean} [triggerRefresh=true]
   */
  function setMode(mode, triggerRefresh = true) {
    if (!state.map) return;
    state.mode = mode;

    // Remove all exclusive overlays
    if (state.layers.heatmap && state.map.hasLayer(state.layers.heatmap)) {
      state.map.removeLayer(state.layers.heatmap);
    }
    if (state.layers.routes && state.map.hasLayer(state.layers.routes)) {
      state.map.removeLayer(state.layers.routes);
    }
    if (state.layers.congestion && state.map.hasLayer(state.layers.congestion)) {
      state.map.removeLayer(state.layers.congestion);
    }
    if (state.layers.trajectory && state.map.hasLayer(state.layers.trajectory)) {
      state.map.removeLayer(state.layers.trajectory);
    }

    // Add back the selected overlay
    if (mode === 'routes') {
      state.layers.routes.addTo(state.map);
    } else if (mode === 'congestion') {
      state.layers.congestion.addTo(state.map);
    } else if (mode === 'trajectory') {
      state.layers.trajectory.addTo(state.map);
    }
    // heatmap is added dynamically when points are loaded

    // Update Mode UI Buttons
    document.querySelectorAll('.traffic-mode-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
    });

    // Toggle trajectory input visibility
    const trajBox = document.getElementById('mapTrajectorySearchBox');
    if (trajBox) {
      trajBox.style.display = mode === 'trajectory' ? 'flex' : 'none';
    }

    updateLegendUI();

    if (triggerRefresh) {
      refreshActiveLayer();
    }
  }

  /**
   * Toggles the Camera base layer on or off.
   *
   * @param {boolean} show
   */
  function toggleCamerasBaseLayer(show) {
    state.showCameras = !!show;
    if (!state.map || !state.layers.cameras) return;

    if (state.showCameras) {
      if (!state.map.hasLayer(state.layers.cameras)) {
        state.layers.cameras.addTo(state.map);
      }
    } else {
      if (state.map.hasLayer(state.layers.cameras)) {
        state.map.removeLayer(state.layers.cameras);
      }
    }
    updateLegendUI();
  }

  /**
   * Builds the query string based on current time filter.
   */
  function getTimeQueryString() {
    const range = calculateTimeRangeQuery(state.timeFilter, state.customFrom, state.customTo);
    if (!range.valid || !range.from) return '';
    return `from=${encodeURIComponent(range.from)}&to=${encodeURIComponent(range.to)}`;
  }

  /**
   * Refreshes data ONLY for the active layer to save CPU and bandwidth.
   */
  async function refreshActiveLayer() {
    if (state.isRefreshing || !state.map) return;
    state.isRefreshing = true;

    setMapStatus('Updating traffic GIS…');

    try {
      const q = getTimeQueryString();
      const queryParam = q ? `?${q}` : '';

      // Always update camera markers if cameras layer is active or in heatmap/cameras mode
      if (state.showCameras || state.mode === 'heatmap' || state.mode === 'cameras') {
        const dRes = await fetch(`/api/analytics/density${queryParam}`).then((r) => r.json()).catch(() => null);
        if (dRes && Array.isArray(dRes.cameras)) {
          state.cameraDataCache = dRes.cameras;
          renderCameraMarkers(dRes.cameras);

          if (state.mode === 'heatmap') {
            renderHeatmap(dRes.cameras);
          }
        }
      }

      // Mode: Route Density
      if (state.mode === 'routes') {
        const rParam = q ? `?limit=50&${q}` : '?limit=50';
        const rRes = await fetch(`/api/analytics/routes${rParam}`).then((r) => r.json()).catch(() => null);
        if (rRes && Array.isArray(rRes.routes)) {
          renderRoutes(rRes.routes);
        } else {
          renderRoutes([]);
        }
      }

      // Mode: Congestion
      if (state.mode === 'congestion') {
        const cRes = await fetch(`/api/analytics/congestion${queryParam}`).then((r) => r.json()).catch(() => null);
        if (cRes && Array.isArray(cRes.routes)) {
          renderCongestion(cRes.routes);
        } else {
          renderCongestion([]);
        }
      }

      // Mode: Trajectory
      if (state.mode === 'trajectory' && state.currentTrackedPlate) {
        await loadTrajectory(state.currentTrackedPlate, false);
      }

      state.lastRefreshTime = new Date();
      setMapStatus(`Active: ${state.mode.toUpperCase()} · ${state.lastRefreshTime.toTimeString().slice(0, 8)}`);
    } catch (err) {
      console.warn('Traffic GIS refresh error:', err);
      setMapStatus('Update warning — check connection');
    } finally {
      state.isRefreshing = false;
    }
  }

  /**
   * Renders camera node markers with density status colors and rich popups.
   */
  function renderCameraMarkers(camerasList) {
    if (!state.layers.cameras) return;
    state.layers.cameras.clearLayers();

    if (!Array.isArray(camerasList) || camerasList.length === 0) return;

    camerasList.forEach((cam) => {
      const lat = Number(cam.latitude !== undefined ? cam.latitude : cam.lat);
      const lng = Number(cam.longitude !== undefined ? cam.longitude : cam.lng);
      if (isNaN(lat) || isNaN(lng) || lat === 0 || lng === 0) return;

      const style = mapDensityStyle(cam.densityLevel);

      const icon = L.divIcon({
        className: 'traffic-cam-marker',
        html: `
          <div style="position:relative;width:20px;height:20px;display:flex;align-items:center;justify-content:center;">
            <div style="position:absolute;inset:-3px;border-radius:50%;background:${style.glow};animation:marker-pulse 2s infinite ease-out;"></div>
            <div style="width:14px;height:14px;border-radius:50%;background:#0c0d14;border:2px solid ${style.color};box-shadow:0 0 8px ${style.color};"></div>
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });

      const popupHtml = `
        <div style="font-family:'Share Tech Mono',monospace;min-width:200px;line-height:1.45;color:#e8eaf0;">
          <div style="font-weight:700;font-size:13px;color:#00ff41;margin-bottom:4px;border-bottom:1px solid #252840;padding-bottom:2px;">
            ${cam.cameraId || ''} · ${cam.cameraName || 'Monitoring Node'}
          </div>
          <div style="font-size:11px;color:#a0aec0;margin-top:2px;">
            <div><b>Location:</b> ${cam.location || cam.roadName || '—'}</div>
            <div><b>Zone:</b> ${cam.zone || 'Central'}</div>
            <div><b>Traffic Volume:</b> <b style="color:#fff;">${cam.vehicleCount || 0}</b></div>
            <div><b>Unique Vehicles:</b> <b style="color:#00b4ff;">${cam.uniquePlateCount || 0}</b></div>
            <div><b>Density Status:</b> <span class="${style.badgeClass}">${cam.densityLevel || 'LOW'}</span></div>
            <div style="margin-top:3px;font-size:10px;color:var(--text-dim);">● Status: ACTIVE MONITOR</div>
          </div>
        </div>
      `;

      const marker = L.marker([lat, lng], { icon }).bindPopup(popupHtml);
      state.layers.cameras.addLayer(marker);
    });
  }

  /**
   * Renders the Traffic Density Heatmap via Leaflet.heat.
   */
  function renderHeatmap(camerasList) {
    if (!state.map) return;

    if (state.layers.heatmap && state.map.hasLayer(state.layers.heatmap)) {
      state.map.removeLayer(state.layers.heatmap);
    }

    const heatPoints = normalizeHeatmapData(camerasList, state.heatmapMetric || 'vehicleCount');

    if (heatPoints.length === 0) {
      return;
    }

    if (typeof L.heatLayer === 'function') {
      state.layers.heatmap = L.heatLayer(heatPoints, {
        radius: 36,
        blur: 24,
        maxZoom: 16,
        max: 1.0,
        minOpacity: 0.35,
        gradient: {
          0.2: '#00ff41',
          0.4: '#00b4ff',
          0.6: '#ffb400',
          0.8: '#ff6b00',
          1.0: '#ff3b30',
        },
      });

      if (state.mode === 'heatmap') {
        state.layers.heatmap.addTo(state.map);
      }
    } else {
      console.warn('Leaflet.heat plugin not loaded.');
    }
  }

  /**
   * Renders Route Density Corridors with thickness normalized to volume.
   */
  function renderRoutes(routesList) {
    if (!state.layers.routes) return;
    state.layers.routes.clearLayers();

    if (!Array.isArray(routesList) || routesList.length === 0) {
      setMapStatus('No route corridor flow recorded in selected period');
      return;
    }

    const maxVolume = Math.max(...routesList.map((r) => r.vehicleCount || 0), 1);

    // Sort ascending so highest volume routes are drawn on top
    const sorted = [...routesList].sort((a, b) => (a.vehicleCount || 0) - (b.vehicleCount || 0));

    sorted.forEach((r) => {
      const endpoints = resolveRouteEndpoints(r, state.cameraDataCache);
      if (!endpoints) return;

      const weight = normalizeRouteWidth(r.vehicleCount, maxVolume, 2.5, 11);

      const polyline = L.polyline([endpoints.fromCoord, endpoints.toCoord], {
        color: '#00b4ff',
        weight,
        opacity: 0.85,
        lineCap: 'round',
        lineJoin: 'round',
      });

      const speedStr = r.averageSpeedKmh ? `${r.averageSpeedKmh} km/h` : '—';
      const timeStr = r.averageTravelTimeSeconds ? `${r.averageTravelTimeSeconds}s (~${Math.round(r.averageTravelTimeSeconds / 60)}m)` : '—';
      const distStr = r.averageDistanceKm ? `${r.averageDistanceKm} km` : '—';

      const popupHtml = `
        <div style="font-family:'Share Tech Mono',monospace;min-width:210px;line-height:1.45;color:#e8eaf0;">
          <div style="font-weight:700;font-size:13px;color:#00b4ff;margin-bottom:4px;border-bottom:1px solid #252840;padding-bottom:2px;">
            ${r.fromCameraName || r.fromCameraId} → ${r.toCameraName || r.toCameraId}
          </div>
          <div style="font-size:11px;color:#a0aec0;margin-top:2px;">
            <div><b>Corridor Key:</b> <code>${r.route}</code></div>
            <div><b>Vehicle Flow:</b> <b style="color:#fff;">${r.vehicleCount || 0} journeys</b></div>
            <div><b>Estimated Avg Speed:</b> <b style="color:#ffb400;">${speedStr}</b></div>
            <div><b>Avg Travel Time:</b> ${timeStr}</div>
            <div><b>Avg Distance:</b> ${distStr}</div>
          </div>
        </div>
      `;

      polyline.bindPopup(popupHtml);
      state.layers.routes.addLayer(polyline);
    });
  }

  /**
   * Renders Congestion routes color-coded by severity (NORMAL/MODERATE/HIGH/SEVERE).
   */
  function renderCongestion(routesList) {
    if (!state.layers.congestion) return;
    state.layers.congestion.clearLayers();

    if (!Array.isArray(routesList) || routesList.length === 0) {
      setMapStatus('No congestion detected / insufficient journey data');
      return;
    }

    // Sort so SEVERE and HIGH routes are drawn last (highest z-index on top)
    const priority = { NORMAL: 1, MODERATE: 2, HIGH: 3, SEVERE: 4 };
    const sorted = [...routesList].sort((a, b) => (priority[a.level] || 1) - (priority[b.level] || 1));

    sorted.forEach((r) => {
      const endpoints = resolveRouteEndpoints(r, state.cameraDataCache);
      if (!endpoints) return;

      const style = mapCongestionStyle(r.level);

      const polyline = L.polyline([endpoints.fromCoord, endpoints.toCoord], {
        color: style.color,
        weight: style.weight,
        opacity: style.opacity,
        lineCap: 'round',
        lineJoin: 'round',
      });

      const currentSpeedStr = r.currentAverageSpeedKmh ? `${r.currentAverageSpeedKmh} km/h` : '—';
      const baselineSpeedStr = r.baselineAverageSpeedKmh ? `${r.baselineAverageSpeedKmh} km/h` : '—';

      const popupHtml = `
        <div style="font-family:'Share Tech Mono',monospace;min-width:220px;line-height:1.45;color:#e8eaf0;">
          <div style="font-weight:700;font-size:13px;color:${style.color};margin-bottom:4px;border-bottom:1px solid #252840;padding-bottom:2px;">
            🚦 ${r.fromCameraName || r.fromCameraId} → ${r.toCameraName || r.toCameraId}
          </div>
          <div style="font-size:11px;color:#a0aec0;margin-top:2px;">
            <div><b>Congestion Level:</b> <span class="${style.badgeClass}">${r.level}</span></div>
            <div><b>Current Vehicles:</b> <b style="color:#fff;">${r.currentVehicleCount}</b> <small>(Baseline: ${r.baselineVehicleCount})</small></div>
            <div><b>Est. Avg Speed:</b> <b style="color:#ffb400;">${currentSpeedStr}</b> <small>(Baseline: ${baselineSpeedStr})</small></div>
            <div><b>Volume Ratio:</b> ${r.volumeRatio ? `${r.volumeRatio}x` : '—'}</div>
            <div><b>Speed Ratio:</b> ${r.speedRatio ? `${r.speedRatio}x` : '—'}</div>
            <div><b>Baseline Source:</b> <i>${r.baselineSource || 'CONFIGURATION'}</i></div>
            ${r.reason ? `<div style="margin-top:4px;color:#e8eaf0;font-size:10px;font-style:italic;">${r.reason}</div>` : ''}
          </div>
        </div>
      `;

      polyline.bindPopup(popupHtml);
      state.layers.congestion.addLayer(polyline);
    });
  }

  /**
   * Tracks a vehicle plate and draws its trajectory on the City Traffic Map.
   *
   * @param {string} plate
   * @param {boolean} [fitBounds=true]
   */
  async function loadTrajectory(plate, fitBounds = true) {
    if (!plate || !state.map) return;
    const cleanPlate = plate.toUpperCase().replace(/[^A-Z0-9]/g, '');
    state.currentTrackedPlate = cleanPlate;

    setMapStatus(`Fetching trajectory for ${cleanPlate}…`);

    try {
      const res = await fetch(`/api/trajectory/${cleanPlate}`).then((r) => r.json());

      if (!state.layers.trajectory) return;
      state.layers.trajectory.clearLayers();

      if (!res.success || !Array.isArray(res.points) || res.points.length === 0) {
        setMapStatus(`No trajectory sightings found for ${cleanPlate}`);
        return;
      }

      const validPoints = res.points.filter((pt) => pt.lat && pt.lng);
      if (validPoints.length === 0) {
        setMapStatus(`Trajectory points contain missing coordinates`);
        return;
      }

      const boundsPoints = [];

      // 1. Draw observed route: Road-aligned solid polyline
      if (res.roadAligned && res.roadLatLngs && res.roadLatLngs.length > 1) {
        const roadPolyline = L.polyline(res.roadLatLngs, {
          color: '#00b4ff',
          weight: 5,
          opacity: 0.95,
          lineCap: 'round',
          lineJoin: 'round',
        });
        roadPolyline.bindPopup(`
          <div style="font-family:'Share Tech Mono',monospace;font-size:11px;">
            <b style="color:#00b4ff;font-size:13px;">Observed Road Route</b>
            <div>Corridor: <b>${res.roadDistanceKm} km</b> (Road Aligned)</div>
            <div style="color:#00ff41;">Status: ROAD NETWORK ALIGNED</div>
          </div>
        `);
        state.layers.trajectory.addLayer(roadPolyline);
        res.roadLatLngs.forEach((pt) => boundsPoints.push(pt));
      } else {
        const straightCoords = validPoints.map((pt) => [pt.lat, pt.lng]);
        if (straightCoords.length > 1) {
          const fallbackLine = L.polyline(straightCoords, {
            color: '#ff9500',
            weight: 2,
            dashArray: '4, 8',
            opacity: 0.7,
          });
          fallbackLine.bindPopup(`
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;">
              <b style="color:#ff9500;">Approximate camera connection</b>
              <div style="color:#ff3b30;">Road route unavailable</div>
            </div>
          `);
          state.layers.trajectory.addLayer(fallbackLine);
          straightCoords.forEach((pt) => boundsPoints.push(pt));
        }
      }

      // 2. Draw predicted routes (dashed road-aligned lines)
      (res.predictions || []).forEach((pred, pIdx) => {
        const isPrimary = pIdx === 0;
        const predRoute = pred.route;
        if (predRoute && predRoute.latLngs && predRoute.latLngs.length > 1) {
          const predLine = L.polyline(predRoute.latLngs, {
            color: isPrimary ? '#00ff41' : '#ffb400',
            weight: isPrimary ? 4 : 3,
            opacity: isPrimary ? 0.9 : 0.5,
            dashArray: isPrimary ? '8, 8' : '4, 6',
          });
          predLine.bindPopup(`
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;">
              <b style="color:${isPrimary ? '#00ff41' : '#ffb400'};font-size:13px;">${isPrimary ? 'Primary Predicted Route' : 'Alternative Predicted Route'}</b>
              <div>Target: <b>${pred.cameraName}</b> (${pred.location})</div>
              <div>Probability: <b>${Math.round(pred.probability * 100)}%</b> · Road Dist: <b>${predRoute.distanceKm} km</b></div>
              <div style="color:var(--amber);">ETA: <b>${pred.etaFormatted}</b></div>
            </div>
          `);
          state.layers.trajectory.addLayer(predLine);
          predRoute.latLngs.forEach((pt) => boundsPoints.push(pt));
        }

        if (pred.lat && pred.lng) {
          const predIcon = L.divIcon({
            html: `<div style="padding:2px 6px;background:${isPrimary ? '#00ff41' : '#ffb400'};color:#050508;border-radius:10px;font-size:9px;font-weight:800;font-family:'Share Tech Mono',monospace;box-shadow:0 0 10px ${isPrimary ? 'rgba(0,255,65,0.7)' : 'rgba(255,180,0,0.7)'};">${Math.round(pred.probability * 100)}%</div>`,
            className: '',
            iconAnchor: [16, 10],
          });
          const pMarker = L.marker([pred.lat, pred.lng], { icon: predIcon, zIndexOffset: isPrimary ? 900 : 800 })
            .bindPopup(`
              <div style="font-family:'Share Tech Mono',monospace;font-size:11px;min-width:180px;">
                <b style="color:${isPrimary ? '#00ff41' : '#ffb400'};font-size:12px;">Predicted Node: ${pred.cameraName}</b>
                <div>Probability: <b>${Math.round(pred.probability * 100)}%</b> · <span style="color:${pred.confidence === 'HIGH' ? '#00ff41' : (pred.confidence === 'MEDIUM' ? '#ffb400' : '#8899aa')}">${pred.confidence || 'MEDIUM'}</span></div>
                <div style="color:var(--amber);">ETA: <b>${pred.etaFormatted}</b></div>
                ${pred.evidence ? `
                  <div style="border-top:1px solid #252840;margin-top:6px;padding-top:4px;font-size:10px;color:#8899aa;">
                    <b style="color:#00e5ff;">WHY?</b><br/>
                    • <b>${pred.evidence.firstOrderTransitions} / ${pred.evidence.totalOutgoingTransitions}</b> historical transitions<br/>
                    • Road: <b>${pred.evidence.topologyVerified ? '✓ Verified' : 'Unverified'}</b><br/>
                    • Direction: <b>${pred.evidence.directionCompatible ? '✓ Compatible' : 'Reversal'}</b>
                  </div>
                ` : ''}
              </div>
            `);
          state.layers.trajectory.addLayer(pMarker);
          boundsPoints.push([pred.lat, pred.lng]);
        }
      });

      // 3. Add numbered waypoint markers
      validPoints.forEach((pt, idx) => {
        const isCurrent = idx === validPoints.length - 1;
        const waypointIcon = L.divIcon({
          className: 'trajectory-waypoint',
          html: `
            <div style="width:26px;height:26px;border-radius:50%;background:#0c0d14;border:2px solid ${isCurrent ? '#00ff41' : '#00b4ff'};color:${isCurrent ? '#00ff41' : '#00b4ff'};display:flex;align-items:center;justify-content:center;font-weight:800;font-size:11px;font-family:'Share Tech Mono',monospace;box-shadow:0 0 12px ${isCurrent ? 'rgba(0,255,65,0.8)' : 'rgba(0,180,255,0.6)'};">
              ${idx + 1}
            </div>
          `,
          iconSize: [26, 26],
          iconAnchor: [13, 13],
        });

        const timeStr = pt.timestamp ? new Date(pt.timestamp).toLocaleTimeString() : '—';
        const marker = L.marker([pt.lat, pt.lng], { icon: waypointIcon, zIndexOffset: isCurrent ? 1000 : 600 }).bindPopup(`
          <div style="font-family:'Share Tech Mono',monospace;font-size:11px;">
            <div style="color:#00ff41;font-weight:700;">${isCurrent ? '📍 CURRENT LOCATION · ' : ''}Waypoint #${idx + 1} · ${cleanPlate}</div>
            <div>Camera: <b>${pt.cameraId}</b> (${pt.cameraName || ''})</div>
            <div>Time: ${timeStr}</div>
            ${pt.estimatedSpeedKmh ? `<div>Speed to next: <b>${pt.estimatedSpeedKmh} km/h</b></div>` : ''}
          </div>
        `);
        state.layers.trajectory.addLayer(marker);
        boundsPoints.push([pt.lat, pt.lng]);
      });

      if (fitBounds && boundsPoints.length > 0) {
        state.map.fitBounds(boundsPoints, { padding: [40, 40], maxZoom: 15 });
      }

      const displayDist = res.roadDistanceKm !== null && res.roadDistanceKm !== undefined ? res.roadDistanceKm : (res.totalDistanceKm || 0);
      setMapStatus(`Tracking ${cleanPlate} (${validPoints.length} sightings · ${displayDist} km ${res.roadAligned ? 'ROAD ALIGNED' : ''})`);
      updateLegendUI();
    } catch (err) {
      console.warn('Error loading trajectory:', err);
      setMapStatus(`Error tracking ${cleanPlate}`);
    }
  }

  /**
   * Applies preset or custom time filter.
   */
  function setTimeFilter(filterType) {
    state.timeFilter = filterType;

    // Update filter chip UI
    document.querySelectorAll('.time-filter-chip').forEach((chip) => {
      chip.classList.toggle('active', chip.getAttribute('data-filter') === filterType);
    });

    const customPicker = document.getElementById('mapCustomTimePicker');
    if (customPicker) {
      customPicker.style.display = filterType === 'custom' ? 'flex' : 'none';
    }

    if (filterType !== 'custom') {
      refreshActiveLayer();
    }
  }

  /**
   * Applies custom time range from input fields.
   */
  function applyCustomTime() {
    const fromInput = document.getElementById('mapCustomTimeFrom');
    const toInput = document.getElementById('mapCustomTimeTo');
    if (!fromInput || !toInput) return;

    const fromVal = fromInput.value;
    const toVal = toInput.value;

    const validation = calculateTimeRangeQuery('custom', fromVal, toVal);
    if (!validation.valid) {
      alert(`Invalid Time Range: ${validation.error}`);
      return;
    }

    state.customFrom = fromVal;
    state.customTo = toVal;
    state.timeFilter = 'custom';
    refreshActiveLayer();
  }

  /**
   * Searches for a plate entered in map search box.
   */
  function searchPlate() {
    const input = document.getElementById('mapPlateInput');
    if (!input) return;
    const plate = input.value.trim();
    if (!plate) return;

    setMode('trajectory', false);
    loadTrajectory(plate, true);
  }

  /**
   * Starts periodic background refresh.
   */
  function startAutoRefresh() {
    if (state.refreshTimer) clearInterval(state.refreshTimer);
    state.refreshTimer = setInterval(() => {
      // Only refresh if map tab is currently visible
      const analyticsTab = document.getElementById('tab-analytics');
      if (!analyticsTab || analyticsTab.style.display !== 'none') {
        refreshActiveLayer();
      }
    }, state.refreshIntervalMs);
  }

  /**
   * Helper to set status message text.
   */
  function setMapStatus(msg) {
    const el = document.getElementById('trafficMapStatusText');
    if (el) el.textContent = msg;
  }

  /**
   * Toggles fullscreen display for the traffic map card.
   */
  function toggleFullscreen() {
    const card = document.getElementById('trafficMapCard');
    const mapEl = document.getElementById(state.containerId);
    if (!card || !mapEl) return;

    card.classList.toggle('map-fullscreen');
    const isFull = card.classList.contains('map-fullscreen');

    mapEl.style.height = isFull ? 'calc(100vh - 120px)' : '480px';

    setTimeout(() => {
      if (state.map) state.map.invalidateSize();
    }, 150);
  }

  function setHeatmapMetric(metric) {
    state.heatmapMetric = metric || 'vehicleCount';
    if (state.cameraDataCache && state.cameraDataCache.length > 0) {
      renderHeatmap(state.cameraDataCache);
    }
  }

  return {
    init,
    setMode,
    setHeatmapMetric,
    toggleCamerasBaseLayer,
    setTimeFilter,
    applyCustomTime,
    searchPlate,
    loadTrajectory,
    refreshActiveLayer,
    toggleFullscreen,
    getState: () => ({ ...state }),
  };
})();

// ─────────────────────────────────────────────────────────────────────────────
// 3. Dual Export (UMD) for Node.js Testing & Browser Environment
// ─────────────────────────────────────────────────────────────────────────────

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    normalizeHeatmapData,
    normalizeRouteWidth,
    mapCongestionStyle,
    mapDensityStyle,
    resolveRouteEndpoints,
    calculateTimeRangeQuery,
  };
}

if (typeof window !== 'undefined') {
  window.TrafficMap = TrafficMap;
  window.normalizeHeatmapData = normalizeHeatmapData;
  window.normalizeRouteWidth = normalizeRouteWidth;
  window.mapCongestionStyle = mapCongestionStyle;
  window.mapDensityStyle = mapDensityStyle;
  window.resolveRouteEndpoints = resolveRouteEndpoints;
  window.calculateTimeRangeQuery = calculateTimeRangeQuery;
}
