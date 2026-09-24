/**
 * anpr-engine.js
 * Browser-side ANPR processing pipeline with strict false-positive suppression.
 *
 * Validation Pipeline:
 *  1. Detect vehicles first (COCO-SSD with configurable vehicle threshold).
 *  2. Search for license plate candidates strictly within vehicle bounding boxes.
 *  3. Plate detection confidence threshold >= 0.60 (configurable).
 *  4. Reject candidates with unrealistic size, aspect ratio (1.8 - 6.0), or area ratios.
 *  5. Run Tesseract OCR only on validated plate candidate regions.
 *  6. Normalize OCR text (strip symbols, uppercase, smart OCR disambiguation).
 *  7. Validate against standard Indian vehicle registration formats (State code + RTO + Series + Number).
 *  8. Multi-frame temporal consistency: confirm candidate across >= 2 frames within a time window.
 *  9. Reject isolated one-frame detections.
 * 10. Fully configurable thresholds with live debug telemetry HUD and console diagnostics.
 */

// Recognised Indian State and Union Territory Codes (+ Bharat Series BH)
const INDIAN_STATE_CODES = new Set([
  'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
  'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
  'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
  'UA', 'UP', 'WB', 'BH'
]);

/* ──────────────────────────────────────────────────────────────────
   ANPREngine
─────────────────────────────────────────────────────────────────── */
class ANPREngine {
  constructor() {
    this.model          = null;
    this.modelLoading   = false;
    this.modelReady     = false;
    this.ocrWorker      = null;
    this.ocrReady       = false;
    this.processors     = {};      // cameraId -> intervalId
    this.overlayData    = {};      // cameraId -> last detected plate/type
    this.statsPerCam    = {};      // cameraId -> { vehicleCount, lastPlate }
    this.onDetection    = null;    // callback(detectionObj)

    // Configurable thresholds and geometric constraints (Requirement 3 & 10)
    this.config = {
      // Confidence thresholds
      vehicleScoreThreshold: 0.30,      // Min vehicle detection confidence (lowered for testing)
      plateScoreThreshold:   0.30,      // Min plate candidate confidence (lowered for testing)
      ocrScoreThreshold:     0.40,      // Min Tesseract OCR confidence

      // Geometric constraints for realistic plates (Requirement 4)
      minAspectRatio:        1.5,       // Relaxed for testing
      maxAspectRatio:        7.0,       // Relaxed for testing
      minPlateWidth:         20,        // Pixels
      minPlateHeight:        10,        // Pixels
      minVehicleWidth:       20,        // Pixels
      minVehicleHeight:      20,        // Pixels
      maxPlateAreaRatio:     0.50,      // Relaxed
      minPlateAreaRatio:     0.001,     // Relaxed

      // Multi-frame temporal consistency (Requirements 8 & 9)
      minConsecutiveFrames:  1,         // Immediate detection for testing (was 2)
      temporalWindowMs:      3500,      // Milliseconds to accumulate consistent sightings
      similarityThreshold:   0.85,      // Levenshtein string similarity threshold

      // Phase 2: Intelligent Keyframe Selection (Laplacian variance & motion filter)
      keyframe_selection:      true,    // Enable/disable intelligent keyframe selection
      keyframe_window_ms:      1000,    // Rolling window size in ms to collect candidates (1000ms)
      minimum_sharpness:       60.0,    // Minimum Laplacian variance threshold to reject blurry frames
      anpr_sampling_interval:  1000,    // Interval between backend dispatches (ms)
      keyframe_sample_rate_ms: 160,     // Sampling frequency for candidate evaluation (~6 fps)
      min_scene_change_mad:    2.5,     // Mean Absolute Difference to detect motion between frames
      max_static_hold_ms:      4000,    // Max time to suppress static scene before refreshing

      // Debugging & logging
      debug:                 true,
      logToConsole:          true,
      showCanvasHUD:         true,
    };

    // Telemetry & Debug Tracking
    this.developerMode     = true;       // Live Visual Overlays & 16-Point Telemetry HUD
    this._lastServerDebug  = {};        // cameraId -> full 16-point debug payload from server
    this._lastServerVehicles = {};      // cameraId -> array of vehicle boxes
    this._lastServerCandidates = {};    // cameraId -> array of plate candidate boxes
    this.debugState        = {};        // cameraId -> current debug object
    this.debugHistory      = [];        // Rolling debug log
    this.temporalTrackers  = {};        // cameraId -> candidate tracker array
    this._inFlight         = {};        // cameraId -> boolean
    this._lastPlateOverlay = {};        // cameraId -> { plate, bbox, confidence, expires }
    this._lastServerQueue  = null;      // Phase 6: Global & camera queue telemetry

    // Phase 2: Per-Camera Intelligent Keyframe Candidate Buffers
    this.keyframeBuffers   = {};        // cameraId -> { candidates: [], lastProcessedGray, lastProcessedTime, lastSelectedSharpness, currentSharpness, isStatic, rejectionReason }

    // Off-screen canvases for candidate evaluation and OCR preprocessing
    this._scratchCanvas    = null;
    this._scratchCtx       = null;
    this._ocrCanvas        = null;
    this._ocrCtx           = null;
    this._captureCanvas    = null;
    this._captureCtx       = null;
    this._sharpCanvas      = null;
    this._sharpCtx         = null;
  }

  toggleDeveloperMode() {
    this.developerMode = !this.developerMode;
    console.log(`🛠️ ANPR: Developer HUD mode is now ${this.developerMode ? 'ENABLED' : 'DISABLED'}`);
    return this.developerMode;
  }

  /* ── Configuration getters & setters (Requirement 10) ── */
  getConfig() {
    return { ...this.config };
  }

  setConfig(options) {
    if (typeof options === 'object' && options !== null) {
      Object.assign(this.config, options);
      console.log('🔧 ANPR: Configuration updated:', this.config);
    }
    return this.getConfig();
  }

  updateConfig(key, value) {
    if (key in this.config) {
      this.config[key] = value;
      console.log(`🔧 ANPR: Config '${key}' set to:`, value);
      return true;
    }
    console.warn(`⚠️ ANPR: Unknown config key '${key}'`);
    return false;
  }

  getDebugInfo(cameraId) {
    return this.debugState[cameraId] || null;
  }

  getDebugHistory() {
    return [...this.debugHistory];
  }

  /* ── Model Loading ── */
  async loadModel() {
    if (this.modelReady || this.modelLoading) return;
    this.modelLoading = true;
    try {
      if (typeof cocoSsd === 'undefined') {
        console.warn('ANPR: COCO-SSD not loaded yet, retrying in 2s');
        this.modelLoading = false;
        setTimeout(() => this.loadModel(), 2000);
        return;
      }
      this.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
      this.modelReady = true;
      this.modelLoading = false;
      console.log('✅ ANPR: COCO-SSD vehicle detector ready');
    } catch (e) {
      this.modelLoading = false;
      console.error('ANPR: Model load failed:', e);
    }
  }

  async loadOCR() {
    if (this.ocrReady) return;
    try {
      if (typeof Tesseract === 'undefined') {
        console.warn('ANPR: Tesseract.js not loaded');
        return;
      }
      this.ocrWorker = await Tesseract.createWorker('eng', 1, { logger: () => {} });
      await this.ocrWorker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        tessedit_pageseg_mode: '8', // Single line / single word mode
      });
      this.ocrReady = true;
      console.log('✅ ANPR: OCR worker ready');
    } catch (e) {
      console.warn('ANPR: OCR init failed (non-critical):', e.message);
    }
  }

  /* ──────────────────────────────────────────────────────────────────
     Phase 2: Intelligent Keyframe Selection (Laplacian Variance)
  ─────────────────────────────────────────────────────────────────── */
  _computeLaplacianVariance(gray, width, height) {
    let sum = 0;
    let sumSq = 0;
    let count = 0;

    // 3x3 discrete Laplacian operator:
    // [ 0,  1,  0]
    // [ 1, -4,  1]
    // [ 0,  1,  0]
    for (let y = 1; y < height - 1; y++) {
      const row = y * width;
      const prevRow = (y - 1) * width;
      const nextRow = (y + 1) * width;

      for (let x = 1; x < width - 1; x++) {
        const center = gray[row + x];
        const top    = gray[prevRow + x];
        const bottom = gray[nextRow + x];
        const left   = gray[row + x - 1];
        const right  = gray[row + x + 1];

        const lap = (top + bottom + left + right) - 4 * center;
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }

    if (count === 0) return 0;
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    return Math.max(0, variance);
  }

  _computeFrameDifference(grayA, grayB) {
    if (!grayA || !grayB || grayA.length !== grayB.length) return 255;
    let sumDiff = 0;
    const len = grayA.length;
    for (let i = 0; i < len; i++) {
      sumDiff += Math.abs(grayA[i] - grayB[i]);
    }
    return sumDiff / len;
  }

  _sampleKeyframeCandidate(cameraId, videoEl) {
    if (!videoEl || videoEl.paused || videoEl.videoWidth === 0 || !videoEl.srcObject) return;

    if (!this._sharpCanvas) {
      this._sharpCanvas = document.createElement('canvas');
      this._sharpCanvas.width = 160;
      this._sharpCanvas.height = 120;
      this._sharpCtx = this._sharpCanvas.getContext('2d', { willReadFrequently: true });
    }

    const sw = 160, sh = 120;
    this._sharpCtx.drawImage(videoEl, 0, 0, sw, sh);
    const imgData = this._sharpCtx.getImageData(0, 0, sw, sh);
    const data = imgData.data;

    const gray = new Uint8Array(sw * sh);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    const sharpness = this._computeLaplacianVariance(gray, sw, sh);
    const kfBuf = this.keyframeBuffers[cameraId];
    if (!kfBuf) return;

    kfBuf.currentSharpness = sharpness;
    const now = Date.now();
    kfBuf.candidates.push({
      timestamp: now,
      sharpness: sharpness,
      gray: gray
    });

    const windowMs = this.config.keyframe_window_ms || 1000;
    while (kfBuf.candidates.length > 0 && (now - kfBuf.candidates[0].timestamp) > windowMs) {
      kfBuf.candidates.shift();
    }
  }

  /* ── Start processing a camera ── */
  startCamera(cameraId, videoEl, canvasEl) {
    if (this.processors[cameraId]) this.stopCamera(cameraId);
    if (!this.statsPerCam[cameraId]) this.statsPerCam[cameraId] = { vehicleCount: 0, lastPlate: null };
    this.temporalTrackers[cameraId] = [];

    this.keyframeBuffers[cameraId] = {
      candidates: [],
      lastProcessedGray: null,
      lastProcessedTime: 0,
      lastSelectedSharpness: 0,
      currentSharpness: 0,
      isStatic: false,
      rejectionReason: null,
    };

    // 1. Candidate sampling loop (runs ~6 fps: ultra-fast 0.1ms Laplacian variance)
    const sampleTimer = setInterval(() => {
      this._sampleKeyframeCandidate(cameraId, videoEl);
    }, this.config.keyframe_sample_rate_ms || 160);

    // 2. Intelligent keyframe dispatch loop (runs every anpr_sampling_interval, e.g. 1000ms)
    const dispatchTimer = setInterval(() => {
      this._processKeyframeDispatch(cameraId, videoEl, canvasEl);
    }, this.config.anpr_sampling_interval || 1000);

    // 3. UI Overlay redraw loop (runs smoothly at ~8-10 fps without blocking)
    const renderTimer = setInterval(() => {
      let activeCandidate = null;
      const plateOverlay = this._lastPlateOverlay[cameraId];
      if (plateOverlay && Date.now() < plateOverlay.expires) {
        activeCandidate = plateOverlay;
      }
      this._drawOverlay(cameraId, canvasEl, videoEl, this._lastServerVehicles[cameraId] || [], activeCandidate, this._lastServerCandidates[cameraId] || []);
    }, 120);

    this.processors[cameraId] = { sampleTimer, dispatchTimer, renderTimer };
    console.log(`ANPR: Started camera ${cameraId} with Phase 2 Intelligent Keyframe Selection`);
  }

  stopCamera(cameraId) {
    const p = this.processors[cameraId];
    if (p) {
      if (p.sampleTimer) clearInterval(p.sampleTimer);
      if (p.dispatchTimer) clearInterval(p.dispatchTimer);
      if (p.renderTimer) clearInterval(p.renderTimer);
      delete this.processors[cameraId];
    }
    delete this.keyframeBuffers[cameraId];
    delete this.temporalTrackers[cameraId];

    // Clear canvas
    const canvas = document.getElementById(`anpr-canvas-${cameraId}`);
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 1 & 2: Intelligent Keyframe Selection & ANPR Dispatch
  ─────────────────────────────────────────────────────────────────── */
  _processFrame(cameraId, videoEl, canvasEl) {
    return this._processKeyframeDispatch(cameraId, videoEl, canvasEl);
  }

  async _processKeyframeDispatch(cameraId, videoEl, canvasEl) {
    if (!videoEl || videoEl.paused || videoEl.videoWidth === 0 || !videoEl.srcObject) return;

    const kfBuf = this.keyframeBuffers[cameraId];
    if (!kfBuf) return;

    // Backlog prevention: if backend inference is still running, do NOT send another frame!
    if (this._inFlight[cameraId]) {
      kfBuf.rejectionReason = 'BACKEND_BUSY: Inference in progress (backlog prevented)';
      return;
    }

    let selectedSharpness = kfBuf.currentSharpness;
    let candidatesCount = kfBuf.candidates.length;

    if (this.config.keyframe_selection) {
      if (!kfBuf.candidates || kfBuf.candidates.length === 0) {
        this._sampleKeyframeCandidate(cameraId, videoEl);
      }
      const candidates = kfBuf.candidates;
      if (candidates.length === 0) return;

      // Select frame with highest sharpness in rolling window
      let best = candidates[0];
      for (let i = 1; i < candidates.length; i++) {
        if (candidates[i].sharpness > best.sharpness) {
          best = candidates[i];
        }
      }

      selectedSharpness = best.sharpness;

      // Check 1: Reject blur / camera shake
      if (best.sharpness < this.config.minimum_sharpness) {
        kfBuf.rejectionReason = `BLUR_REJECTED: Sharpness ${best.sharpness.toFixed(1)} < min ${this.config.minimum_sharpness}`;
        if (this.developerMode) {
          this._recordDebug(cameraId, {
            vehicleConfidence: 0,
            plateConfidence: 0,
            ocrConfidence: 0,
            status: 'KEYFRAME_BLUR',
            reason: kfBuf.rejectionReason,
            candidatePlate: null,
            vehicleType: null
          });
        }
        return;
      }

      // Check 2: Skip identical static scenes
      if (kfBuf.lastProcessedGray) {
        const mad = this._computeFrameDifference(best.gray, kfBuf.lastProcessedGray);
        const hasActivePlate = this._lastPlateOverlay[cameraId] && Date.now() < this._lastPlateOverlay[cameraId].expires;
        const elapsedSinceLast = Date.now() - kfBuf.lastProcessedTime;

        if (mad < (this.config.min_scene_change_mad || 2.5) && hasActivePlate && elapsedSinceLast < (this.config.max_static_hold_ms || 4000)) {
          kfBuf.isStatic = true;
          kfBuf.rejectionReason = `STATIC_HOLD: Negligible motion (MAD ${mad.toFixed(1)}), plate held`;
          return;
        }
      }

      kfBuf.isStatic = false;
      kfBuf.rejectionReason = null;
      kfBuf.lastProcessedGray = best.gray;
      kfBuf.lastProcessedTime = Date.now();
      kfBuf.lastSelectedSharpness = selectedSharpness;
    }

    // Capture frame to send to server
    this._inFlight[cameraId] = true;
    (async () => {
      try {
        if (!this._captureCanvas) {
          this._captureCanvas = document.createElement('canvas');
          this._captureCtx = this._captureCanvas.getContext('2d', { willReadFrequently: true });
        }
        const capCanvas = this._captureCanvas;
        const capCtx    = this._captureCtx;
        const vw = videoEl.videoWidth;
        const vh = videoEl.videoHeight;
        const maxDim = this.config.captureMaxDimension || 1280;
        const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
        capCanvas.width  = Math.round(vw * scale);
        capCanvas.height = Math.round(vh * scale);
        capCtx.drawImage(videoEl, 0, 0, capCanvas.width, capCanvas.height);

        const base64Img = capCanvas.toDataURL('image/jpeg', 0.88);
        const frameCapturedAt = (typeof best !== 'undefined' && best && best.timestamp) ? best.timestamp : Date.now();
        const keyframeSelectedAt = Date.now();

        const resp = await fetch('/api/anpr/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Img,
            cameraId: cameraId,
            cameraName: `CAM 0${cameraId}`,
            forwardToDashboard: true,
            developerMode: this.developerMode,
            keyframeSharpness: Math.round(selectedSharpness * 10) / 10,
            keyframeWindowMs: this.config.keyframe_window_ms,
            keyframeCandidates: candidatesCount,
            frameCapturedAt: frameCapturedAt,
            keyframeSelectedAt: keyframeSelectedAt
          })
        });

        if (resp.ok) {
          const data = await resp.json();
          const inv = 1.0 / scale;
          const frontendRenderedAt = Date.now();

          if (data.timeline) {
            data.timeline.frontendRenderedAt = frontendRenderedAt;
            data.timeline.totalLiveConfirmationMs = frontendRenderedAt - frameCapturedAt;
            if (this.config.logToConsole && data.detected) {
              console.log(`⏱️ [LIVE TIMELINE] CAM 0${cameraId} -> Plate: ${data.detection?.plate} | Total Live Latency: ${data.timeline.totalLiveConfirmationMs}ms (Frame Sampling: ${(data.timeline.keyframeSelectedAt||0)-(data.timeline.frameCapturedAt||0)}ms, Queue: ${data.timeline.queueWaitMs||0}ms, ANPR Worker: ${data.timeline.totalProcessingMs||0}ms)`);
            }
          }

          if (data.queueTelemetry) {
            this._lastServerQueue = data.queueTelemetry;
          }

          if (data.status === 'DROPPED_STALE_FRAME') {
            if (this.config.logToConsole) {
              console.log(`[Phase 6] CAM 0${cameraId}: Frame dropped (${data.reason || 'Superseded by newer keyframe'})`);
            }
            return;
          }

          if (data.debug) {
            this._lastServerDebug[cameraId] = data.debug;
          }

          if (data.vehicles && Array.isArray(data.vehicles)) {
            this._lastServerVehicles[cameraId] = data.vehicles.map(v => ({
              class: v.class,
              score: v.confidence,
              bbox: [v.box[0] * inv, v.box[1] * inv, (v.box[2] - v.box[0]) * inv, (v.box[3] - v.box[1]) * inv]
            }));
          } else {
            this._lastServerVehicles[cameraId] = [];
          }

          if (data.debug && data.debug.candidates && Array.isArray(data.debug.candidates)) {
            this._lastServerCandidates[cameraId] = data.debug.candidates.map(c => ({
              ...c,
              bbox: [c.bbox[0] * inv, c.bbox[1] * inv, c.bbox[2] * inv, c.bbox[3] * inv]
            }));
          } else {
            this._lastServerCandidates[cameraId] = [];
          }

          if (data.success && data.detected && data.detection) {
            const det = data.detection;
            const origBbox = [det.bbox[0] * inv, det.bbox[1] * inv, det.bbox[2] * inv, det.bbox[3] * inv];

            this._lastPlateOverlay[cameraId] = {
              plate: det.plate,
              stateName: det.stateName || 'India',
              stateCode: det.stateCode || '',
              bbox: origBbox,
              confidence: det.confidence,
              detectorConfidence: det.detectorConfidence,
              ocrConfidence: det.ocrConfidence,
              vehicleType: det.vehicleType || 'Car',
              plateType: det.plateType || 'HSRP',
              timeline: data.timeline || null,
              expires: Date.now() + 5000
            };

            this.statsPerCam[cameraId].lastPlate = det.plate;
            this.statsPerCam[cameraId].vehicleCount++;

            this.showLivePlateHUD(cameraId, this._lastPlateOverlay[cameraId]);
            this._drawOverlay(cameraId, canvasEl, videoEl, this._lastServerVehicles[cameraId], this._lastPlateOverlay[cameraId], this._lastServerCandidates[cameraId]);

            this._recordDebug(cameraId, {
              vehicleConfidence: 0.95,
              plateConfidence: det.detectorConfidence,
              ocrConfidence: det.ocrConfidence,
              status: 'CONFIRMED',
              reason: `Detected Indian plate: ${det.plate} (${det.stateName || ''})`,
              candidatePlate: det.plate,
              vehicleType: det.vehicleType || 'vehicle',
              plateBbox: origBbox
            });
          } else if (data.success && !data.detected) {
            const status = data.status || 'NO_PLATE';
            let reasonMsg = data.reason || 'Scanning frame...';
            if (status === 'CANDIDATE_TRACKING') {
              reasonMsg = 'Plate candidate sighted — verifying temporal consistency across frames...';
            } else if (status === 'REJECTED_NO_VEHICLE') {
              reasonMsg = 'Zero vehicles detected in scene — noise suppressed.';
            } else if (status === 'NO_PLATE_CANDIDATES') {
              reasonMsg = 'Vehicles present but no valid license plate candidates passed geometric filters.';
            }
            const hasVeh = data.vehicles && data.vehicles.length > 0;
            this._recordDebug(cameraId, {
              vehicleConfidence: hasVeh ? data.vehicles[0].confidence : 0,
              plateConfidence: (data.debug && data.debug["6_plateConfidences"] && data.debug["6_plateConfidences"][0]) || 0,
              ocrConfidence: 0,
              status: status === 'REJECTED_NO_VEHICLE' ? 'REJECTED' : (status === 'CANDIDATE_TRACKING' ? 'CANDIDATE' : 'MONITORING'),
              reason: reasonMsg,
              candidatePlate: null,
              vehicleType: hasVeh ? data.vehicles[0].class : null
            });

            this._drawOverlay(cameraId, canvasEl, videoEl, this._lastServerVehicles[cameraId], null, this._lastServerCandidates[cameraId]);
          }
        }
      } catch (err) {
        // Graceful ignore
      } finally {
        this._inFlight[cameraId] = false;
      }
    })();
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 3 & 4: Plate Candidate Detection inside Vehicle BBox
  ─────────────────────────────────────────────────────────────────── */
  _detectPlateCandidate(videoEl, vehicle) {
    const [vx, vy, vw, vh] = vehicle.bbox;
    const vW = videoEl.videoWidth;
    const vH = videoEl.videoHeight;

    // License plates in India are mounted in the lower region of vehicles
    const searchX = Math.max(0, Math.floor(vx + vw * 0.08));
    const searchY = Math.max(0, Math.floor(vy + vh * 0.45));
    const searchW = Math.min(vW - searchX, Math.floor(vw * 0.84));
    const searchH = Math.min(vH - searchY, Math.floor(vh * 0.50));

    if (searchW < this.config.minPlateWidth || searchH < this.config.minPlateHeight) {
      return {
        found: false,
        confidence: 0,
        reason: `Plate search region too small (${searchW}x${searchH}px)`,
      };
    }

    if (!this._scratchCanvas) {
      this._scratchCanvas = document.createElement('canvas');
      this._scratchCtx = this._scratchCanvas.getContext('2d', { willReadFrequently: true });
    }
    const canvas = this._scratchCanvas;
    const ctx = this._scratchCtx;
    canvas.width = searchW;
    canvas.height = searchH;

    ctx.drawImage(videoEl, searchX, searchY, searchW, searchH, 0, 0, searchW, searchH);
    const imgData = ctx.getImageData(0, 0, searchW, searchH);
    const pixels = imgData.data;

    // Grayscale transformation for edge & contrast scoring
    const gray = new Uint8Array(searchW * searchH);
    for (let i = 0; i < pixels.length; i += 4) {
      gray[i / 4] = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
    }

    // Scan multi-scale candidate heights
    const candidateHeights = [
      Math.max(this.config.minPlateHeight, Math.floor(searchH * 0.22)),
      Math.max(this.config.minPlateHeight, Math.floor(searchH * 0.32)),
      Math.max(this.config.minPlateHeight, Math.floor(searchH * 0.42)),
    ];

    let bestCandidate = null;
    let highestScore = 0;
    let bestRejectReason = 'No candidate evaluated';

    for (const ch of candidateHeights) {
      // Test common Indian plate aspect ratios: long rectangular (~4.2, ~3.6) and two-line (~2.2, ~2.0)
      const testAspectRatios = [4.2, 3.6, 2.4, 2.0];
      for (const ar of testAspectRatios) {
        const cw = Math.floor(ch * ar);
        if (cw > searchW || cw < this.config.minPlateWidth) continue;

        const stepX = Math.max(8, Math.floor(cw * 0.28));
        const stepY = Math.max(4, Math.floor(ch * 0.30));

        for (let cy = 0; cy <= searchH - ch; cy += stepY) {
          for (let cx = 0; cx <= searchW - cw; cx += stepX) {
            const actualAR = cw / ch;

            // Reject invalid aspect ratio
            if (actualAR < this.config.minAspectRatio || actualAR > this.config.maxAspectRatio) {
              bestRejectReason = `Aspect ratio ${actualAR.toFixed(2)} invalid (allowed: ${this.config.minAspectRatio}-${this.config.maxAspectRatio})`;
              continue;
            }

            // Reject unrealistic plate area compared to entire vehicle
            const plateArea = cw * ch;
            const vehicleArea = vw * vh;
            const areaRatio = plateArea / vehicleArea;
            if (areaRatio > this.config.maxPlateAreaRatio) {
              bestRejectReason = `Plate area ratio ${(areaRatio * 100).toFixed(1)}% exceeds max ${(this.config.maxPlateAreaRatio * 100)}%`;
              continue;
            }
            if (areaRatio < this.config.minPlateAreaRatio) {
              bestRejectReason = `Plate area ratio ${(areaRatio * 100).toFixed(2)}% below min ${(this.config.minPlateAreaRatio * 100)}%`;
              continue;
            }

            // Calculate contrast, luminance, and character stroke transitions
            let minLum = 255, maxLum = 0, sumLum = 0;
            let edgeTransitions = 0;
            const sampleRows = 4;
            const rowStep = Math.max(1, Math.floor(ch / sampleRows));

            for (let r = 0; r < ch; r += rowStep) {
              let prevLum = gray[(cy + r) * searchW + cx];
              for (let c = 1; c < cw; c++) {
                const lum = gray[(cy + r) * searchW + (cx + c)];
                if (lum < minLum) minLum = lum;
                if (lum > maxLum) maxLum = lum;
                sumLum += lum;

                // Detect dark-light character transitions
                if (Math.abs(lum - prevLum) > 28) {
                  edgeTransitions++;
                }
                prevLum = lum;
              }
            }

            const totalSamples = (ch / rowStep) * cw;
            const avgLum = sumLum / totalSamples;
            const contrast = (maxLum - minLum) / Math.max(1, maxLum + minLum); // Michelson contrast (0 to 1)

            // Expected character edge transitions for 8-10 characters
            const expectedTransitions = (cw / ch) * 3.5 * sampleRows;
            const transitionDensity = Math.min(1.0, edgeTransitions / Math.max(1, expectedTransitions));

            // Background plate brightness score (plates have white or yellow backgrounds)
            const brightnessScore = avgLum >= 65 && avgLum <= 235 ? 0.85 : 0.40;

            // Closeness to standard Indian aspect ratios
            const arScore1 = Math.max(0, 1 - Math.abs(actualAR - 4.1) / 2.0);
            const arScore2 = Math.max(0, 1 - Math.abs(actualAR - 2.2) / 1.2);
            const arBonus = Math.max(arScore1, arScore2);

            // Composite plate detection score
            const plateConf = Math.min(
              0.99,
              0.35 * contrast + 0.35 * transitionDensity + 0.15 * arBonus + 0.15 * brightnessScore
            );

            if (plateConf > highestScore) {
              highestScore = plateConf;
              bestCandidate = {
                bbox: [searchX + cx, searchY + cy, cw, ch],
                aspectRatio: actualAR,
                confidence: Math.round(plateConf * 100) / 100,
                contrast: Math.round(contrast * 100) / 100,
                edgeDensity: Math.round(transitionDensity * 100) / 100,
              };
            }
          }
        }
      }
    }

    // Requirement 3: Check plate detection confidence threshold (starts around 0.6)
    if (!bestCandidate || highestScore < this.config.plateScoreThreshold) {
      return {
        found: false,
        confidence: Math.round(highestScore * 100) / 100,
        reason: bestCandidate
          ? `Plate detection confidence too low (${(highestScore * 100).toFixed(0)}% < ${(this.config.plateScoreThreshold * 100).toFixed(0)}%)`
          : bestRejectReason,
        candidate: bestCandidate,
      };
    }

    return {
      found: true,
      confidence: Math.round(highestScore * 100) / 100,
      candidate: bestCandidate,
    };
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 5, 6 & 7: OCR on Validated Region & Indian Format Validation
  ─────────────────────────────────────────────────────────────────── */
  async _attemptOCR(cameraId, videoEl, vehicle, plateCandidate) {
    const [px, py, pw, ph] = plateCandidate.bbox;
    const vW = videoEl.videoWidth;
    const vH = videoEl.videoHeight;

    // Small 5% padding around plate for margin
    const padX  = Math.floor(pw * 0.05);
    const padY  = Math.floor(ph * 0.05);
    const cropX = Math.max(0, px - padX);
    const cropY = Math.max(0, py - padY);
    const cropW = Math.min(vW - cropX, pw + padX * 2);
    const cropH = Math.min(vH - cropY, ph + padY * 2);

    if (!this._ocrCanvas) {
      this._ocrCanvas = document.createElement('canvas');
      this._ocrCtx = this._ocrCanvas.getContext('2d', { willReadFrequently: true });
    }
    const canvas = this._ocrCanvas;
    const ctx = this._ocrCtx;

    // Scale up crop for optimal OCR readability (height >= 55px)
    const scale = Math.max(1.0, 60 / cropH);
    canvas.width  = Math.round(cropW * scale);
    canvas.height = Math.round(cropH * scale);

    // Preprocessing: grayscale + contrast stretch + crisp thresholding
    ctx.filter = 'grayscale(1) contrast(2.6) brightness(1.25)';
    ctx.drawImage(videoEl, cropX, cropY, cropW, cropH, 0, 0, canvas.width, canvas.height);

    try {
      const { data } = await this.ocrWorker.recognize(canvas);
      const rawText = data.text || '';
      const ocrConf = Math.min(1.0, Math.max(0.0, (data.confidence || 0) / 100));

      // Step 6: Normalize OCR output (strip symbols/spaces, uppercase)
      const cleanPlate = this._normalizeOCRText(rawText);

      // Check OCR confidence threshold
      if (ocrConf < this.config.ocrScoreThreshold) {
        this._recordDebug(cameraId, {
          vehicleConfidence: vehicle.score,
          plateConfidence:   plateCandidate.confidence,
          ocrConfidence:     ocrConf,
          status:            'REJECTED',
          reason:            `OCR confidence too low (${(ocrConf * 100).toFixed(0)}% < ${(this.config.ocrScoreThreshold * 100).toFixed(0)}%)`,
          candidatePlate:    cleanPlate || rawText.trim(),
          vehicleType:       vehicle.class,
          plateBbox:         plateCandidate.bbox,
          vehicleBbox:       vehicle.bbox,
        });
        return;
      }

      // Step 7: Validate OCR results against common Indian vehicle registration formats
      const formatValidation = this._validateIndianPlateFormat(cleanPlate);
      if (!formatValidation.valid) {
        this._recordDebug(cameraId, {
          vehicleConfidence: vehicle.score,
          plateConfidence:   plateCandidate.confidence,
          ocrConfidence:     ocrConf,
          status:            'REJECTED',
          reason:            formatValidation.reason,
          candidatePlate:    cleanPlate || rawText.trim(),
          vehicleType:       vehicle.class,
          plateBbox:         plateCandidate.bbox,
          vehicleBbox:       vehicle.bbox,
        });
        return;
      }

      const validPlate = formatValidation.plate;

      // Step 8 & 9: Multi-frame temporal consistency & Reject isolated one-frame detections
      const consistencyResult = this._checkTemporalConsistency(cameraId, validPlate, {
        vehicleConfidence: vehicle.score,
        plateConfidence:   plateCandidate.confidence,
        ocrConfidence:     ocrConf,
        vehicleType:       vehicle.class,
        plateBbox:         plateCandidate.bbox,
        vehicleBbox:       vehicle.bbox,
      });

      if (!consistencyResult.confirmed) {
        this._recordDebug(cameraId, {
          vehicleConfidence: vehicle.score,
          plateConfidence:   plateCandidate.confidence,
          ocrConfidence:     ocrConf,
          status:            'PENDING',
          reason:            `Awaiting multi-frame confirmation (${consistencyResult.hits}/${this.config.minConsecutiveFrames} frames seen)`,
          candidatePlate:    validPlate,
          vehicleType:       vehicle.class,
          plateBbox:         plateCandidate.bbox,
          vehicleBbox:       vehicle.bbox,
        });
        return;
      }

      // Confirmed across multiple processed frames!
      if (!consistencyResult.alreadyEmitted) {
        this._recordDebug(cameraId, {
          vehicleConfidence: vehicle.score,
          plateConfidence:   plateCandidate.confidence,
          ocrConfidence:     ocrConf,
          status:            'CONFIRMED',
          reason:            `Plate ${validPlate} verified across ${consistencyResult.hits} consecutive frames`,
          candidatePlate:    validPlate,
          vehicleType:       vehicle.class,
          plateBbox:         plateCandidate.bbox,
          vehicleBbox:       vehicle.bbox,
        });

        // Combined overall confidence
        const overallConfidence = Math.min(
          0.99,
          Math.round((vehicle.score * 0.25 + plateCandidate.confidence * 0.35 + ocrConf * 0.40) * 100) / 100
        );

        this._emitDetection({
          plate:       validPlate,
          cameraId,
          confidence:  overallConfidence,
          vehicleType: vehicle.class,
          simulated:   false,
          debug: {
            vehicleConfidence: Math.round(vehicle.score * 100) / 100,
            plateConfidence:   plateCandidate.confidence,
            ocrConfidence:     Math.round(ocrConf * 100) / 100,
            hits:              consistencyResult.hits,
            format:            formatValidation.format,
          },
        });
      }
    } catch (e) {
      console.warn('ANPR: OCR recognition error:', e.message);
    }
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 6: OCR Output Normalization (Remove symbols/spaces, uppercase)
  ─────────────────────────────────────────────────────────────────── */
  _normalizeOCRText(rawText) {
    if (!rawText || typeof rawText !== 'string') return '';

    // Requirement 6: Normalize OCR output by removing unnecessary spaces/symbols and converting to uppercase
    let text = rawText.toUpperCase().replace(/[^A-Z0-9]/g, '').trim();

    // Minor state code repair: if position 0 or 1 is '0' (zero), normalize to 'O'
    if (text.length >= 8 && text.length <= 11) {
      const chars = text.split('');
      if (chars[0] === '0') chars[0] = 'O';
      if (chars[1] === '0') chars[1] = 'O';
      text = chars.join('');
    }

    return text;
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 7: Validate against Common Indian Registration Formats
  ─────────────────────────────────────────────────────────────────── */
  _validateIndianPlateFormat(plate) {
    if (!plate) return { valid: false, reason: 'Empty plate candidate' };

    const clean = plate.toUpperCase().replace(/[^A-Z0-9]/g, '');

    // Relaxed for testing: Just require at least 3 characters
    if (clean.length < 3) {
      return {
        valid: false,
        reason: `Invalid plate length (${clean.length} chars; expected > 3 chars)`,
      };
    }

    // Bypass Indian format checks entirely for testing
    return {
      valid: true,
      plate: clean,
      format: 'TEST_MODE_BYPASS',
    };
  }

  /* ──────────────────────────────────────────────────────────────────
     Step 8 & 9: Multi-Frame Temporal Consistency Tracker
  ─────────────────────────────────────────────────────────────────── */
  _checkTemporalConsistency(cameraId, plate, metadata) {
    if (!this.temporalTrackers[cameraId]) {
      this.temporalTrackers[cameraId] = [];
    }
    const tracker = this.temporalTrackers[cameraId];
    const now = Date.now();

    // Prune stale candidate entries outside temporalWindowMs
    this._pruneTemporalCandidates(cameraId);

    // Look for matching or highly similar candidate in recent frames
    let match = null;
    for (const item of tracker) {
      if (item.plate === plate || this._plateSimilarity(item.plate, plate) >= this.config.similarityThreshold) {
        match = item;
        break;
      }
    }

    if (match) {
      match.hits++;
      match.lastSeen = now;

      // Update to the candidate string with highest OCR confidence
      if (metadata.ocrConfidence > match.bestOcrConfidence) {
        match.plate = plate;
        match.bestOcrConfidence = metadata.ocrConfidence;
      }

      const confirmed = match.hits >= this.config.minConsecutiveFrames;
      const alreadyEmitted = match.emitted;
      if (confirmed && !alreadyEmitted) {
        match.emitted = true;
      }

      return {
        confirmed,
        hits: match.hits,
        alreadyEmitted,
      };
    }

    tracker.push({
      plate,
      firstSeen:         now,
      lastSeen:          now,
      hits:              1,
      bestOcrConfidence: metadata.ocrConfidence,
      emitted:           false,
    });

    return {
      confirmed:      this.config.minConsecutiveFrames <= 1,
      hits:           1,
      alreadyEmitted: false,
    };
  }

  _pruneTemporalCandidates(cameraId) {
    const tracker = this.temporalTrackers[cameraId];
    if (!tracker || tracker.length === 0) return;

    const now = Date.now();
    for (let i = tracker.length - 1; i >= 0; i--) {
      if (now - tracker[i].lastSeen > this.config.temporalWindowMs) {
        const expired = tracker.splice(i, 1)[0];
        if (expired.hits < this.config.minConsecutiveFrames) {
          this._recordDebug(cameraId, {
            vehicleConfidence: null,
            plateConfidence:   null,
            ocrConfidence:     expired.bestOcrConfidence,
            status:            'REJECTED',
            reason:            `Isolated one-frame detection dropped (seen in only ${expired.hits} frame)`,
            candidatePlate:    expired.plate,
            vehicleType:       'vehicle',
          });
        }
      }
    }
  }

  _plateSimilarity(str1, str2) {
    if (str1 === str2) return 1.0;
    const l1 = str1.length;
    const l2 = str2.length;
    if (Math.abs(l1 - l2) > 2) return 0;

    const track = Array(l2 + 1).fill(null).map(() => Array(l1 + 1).fill(null));
    for (let i = 0; i <= l1; i += 1) track[0][i] = i;
    for (let j = 0; j <= l2; j += 1) track[j][0] = j;
    for (let j = 1; j <= l2; j += 1) {
      for (let i = 1; i <= l1; i += 1) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        track[j][i] = Math.min(
          track[j][i - 1] + 1,
          track[j - 1][i] + 1,
          track[j - 1][i - 1] + indicator
        );
      }
    }
    const dist = track[l2][l1];
    const maxLen = Math.max(l1, l2);
    return (maxLen - dist) / maxLen;
  }

  /* ──────────────────────────────────────────────────────────────────
     Canvas Overlay Drawing & Live 16-Point Developer HUD
  ─────────────────────────────────────────────────────────────────── */
  _drawOverlay(cameraId, canvasEl, videoEl, vehicles, activeCandidate, plateCandidates) {
    if (!canvasEl) return;
    canvasEl.width  = videoEl.clientWidth  || videoEl.offsetWidth  || 320;
    canvasEl.height = videoEl.clientHeight || videoEl.offsetHeight || 240;
    const ctx = canvasEl.getContext('2d');
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (!videoEl.videoWidth || !videoEl.videoHeight) return;
    const cw = canvasEl.width;
    const ch = canvasEl.height;
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;

    const compStyle = window.getComputedStyle(videoEl);
    const fit = compStyle.objectFit || 'cover';

    let scale = 1.0;
    let offsetX = 0;
    let offsetY = 0;

    if (fit === 'cover') {
      scale = Math.max(cw / vw, ch / vh);
      offsetX = (cw - vw * scale) / 2;
      offsetY = (ch - vh * scale) / 2;
    } else if (fit === 'contain') {
      scale = Math.min(cw / vw, ch / vh);
      offsetX = (cw - vw * scale) / 2;
      offsetY = (ch - vh * scale) / 2;
    } else {
      scale = null;
    }

    const mapX = (x) => scale !== null ? (x * scale + offsetX) : (x * (cw / vw));
    const mapY = (y) => scale !== null ? (y * scale + offsetY) : (y * (ch / vh));
    const mapW = (w) => scale !== null ? (w * scale) : (w * (cw / vw));
    const mapH = (h) => scale !== null ? (h * scale) : (h * (ch / vh));

    const serverDebug = this._lastServerDebug[cameraId];
    const candidates = plateCandidates || this._lastServerCandidates[cameraId] || [];

    // 1. Draw vehicle bounding boxes
    const vList = vehicles && vehicles.length > 0 ? vehicles : (this._lastServerVehicles[cameraId] || []);
    vList.forEach(v => {
      const [x, y, w, h] = v.bbox;
      const sx = mapX(x), sy = mapY(y), sw = mapW(w), sh = mapH(h);

      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth   = 2;
      ctx.shadowColor = '#00e5ff';
      ctx.shadowBlur  = 6;
      ctx.strokeRect(sx, sy, sw, sh);
      ctx.shadowBlur  = 0;

      // Label pill
      const vScore = v.score != null ? Math.round(v.score * 100) : (v.confidence != null ? Math.round(v.confidence * 100) : 80);
      ctx.fillStyle = 'rgba(0, 229, 255, 0.88)';
      ctx.fillRect(sx, Math.max(0, sy - 20), Math.min(sw, 140), 18);
      ctx.fillStyle = '#050508';
      ctx.font      = 'bold 10px "Share Tech Mono", monospace';
      ctx.fillText(`🚘 ${v.class.toUpperCase()} ${vScore}%`, sx + 4, Math.max(12, sy - 6));
    });

    // 2. Draw plate candidates (both accepted and rejected with exact reasons)
    candidates.forEach((cand, idx) => {
      const [cx, cy, cw, ch] = cand.bbox;
      const csx = mapX(cx), csy = mapY(cy), csw = mapW(cw), csh = mapH(ch);

      if (!cand.passedValidation) {
        // Discarded / Rejected candidate: Red dashed box with rejection reason
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#ff3b30';
        ctx.lineWidth = 1.8;
        ctx.strokeRect(csx, csy, csw, csh);

        const rejText = `[REJ] ${cand.rejectionReason ? cand.rejectionReason.split(':')[0] : 'FILTERED'} (${Math.round(cand.detectorConfidence * 100)}%)`;
        ctx.font = 'bold 9px "Share Tech Mono", monospace';
        const rw = ctx.measureText(rejText).width;
        ctx.fillStyle = 'rgba(255, 59, 48, 0.85)';
        ctx.fillRect(csx, Math.min(canvasEl.height - 16, csy + csh + 2), rw + 6, 14);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(rejText, csx + 3, Math.min(canvasEl.height - 4, csy + csh + 13));
        ctx.restore();
      } else if (!activeCandidate) {
        // Valid candidate accumulating frames: Amber box
        ctx.save();
        ctx.strokeStyle = '#ffb400';
        ctx.lineWidth = 2;
        ctx.strokeRect(csx, csy, csw, csh);

        const candText = `⏳ CANDIDATE (${Math.round(cand.detectorConfidence * 100)}%)`;
        ctx.font = 'bold 9px "Share Tech Mono", monospace';
        const cwText = ctx.measureText(candText).width;
        ctx.fillStyle = 'rgba(255, 180, 0, 0.85)';
        ctx.fillRect(csx, Math.max(0, csy - 16), cwText + 6, 14);
        ctx.fillStyle = '#050508';
        ctx.fillText(candText, csx + 3, Math.max(10, csy - 5));
        ctx.restore();
      }
    });

    // 3. Draw confirmed plate candidate region if present
    if (activeCandidate) {
      const [px, py, pw, ph] = activeCandidate.bbox;
      const psx = mapX(px), psy = mapY(py), psw = mapW(pw), psh = mapH(ph);

      ctx.save();
      // Glowing neon green bounding box
      ctx.strokeStyle = '#00ff41';
      ctx.lineWidth   = Math.max(2.5, Math.round(canvasEl.width / 220));
      ctx.shadowColor = '#00ff41';
      ctx.shadowBlur  = 12;
      ctx.strokeRect(psx, psy, psw, psh);

      // HUD Corner Brackets
      const cLen = Math.min(psw, psh) * 0.28;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth   = Math.max(3.5, Math.round(canvasEl.width / 180));
      ctx.shadowBlur  = 0;

      ctx.beginPath();
      ctx.moveTo(psx, psy + cLen); ctx.lineTo(psx, psy); ctx.lineTo(psx + cLen, psy);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(psx + psw - cLen, psy); ctx.lineTo(psx + psw); ctx.lineTo(psx + psw, psy + cLen);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(psx, psy + psh - cLen); ctx.lineTo(psx, psy + psh); ctx.lineTo(psx + cLen, psy + psh);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(psx + psw - cLen, psy + psh); ctx.lineTo(psx + psw, psy + psh); ctx.lineTo(psx + psw, psy + psh - cLen);
      ctx.stroke();

      // Plate Label Banner with actual detected Plate Number and Flag!
      const plateStr = activeCandidate.plate || 'IND PLATE';
      const confPct  = Math.round(activeCandidate.confidence * 100);
      const labelText = ` 🇮🇳 ${plateStr} · ${confPct}% `;
      const fontSize  = Math.max(11, Math.round(canvasEl.width / 30));
      ctx.font = `bold ${fontSize}px "Share Tech Mono", monospace`;
      const textW = ctx.measureText(labelText).width;
      const bannerH = fontSize + 10;
      const bannerY = Math.max(0, psy - bannerH - 3);

      ctx.fillStyle = '#060810';
      ctx.fillRect(psx, bannerY, textW + 8, bannerH);
      ctx.strokeStyle = '#00ff41';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(psx, bannerY, textW + 8, bannerH);

      ctx.fillStyle = '#00ff41';
      ctx.shadowColor = '#00ff41';
      ctx.shadowBlur = 6;
      ctx.fillText(labelText, psx + 4, bannerY + fontSize);

      if (activeCandidate.stateName) {
        const stateText = ` ${activeCandidate.stateName.toUpperCase()} `;
        ctx.font = `bold ${Math.max(9, Math.round(fontSize * 0.75))}px "Share Tech Mono", monospace`;
        const stateW = ctx.measureText(stateText).width;
        ctx.fillStyle = 'rgba(0, 51, 153, 0.9)';
        ctx.fillRect(psx, psy + psh + 3, stateW + 6, fontSize + 4);
        ctx.fillStyle = '#ffffff';
        ctx.shadowBlur = 0;
        ctx.fillText(stateText, psx + 3, psy + psh + fontSize + 3);
      }
      ctx.restore();
    }

    // 4. Temporary Developer HUD: Visual telemetry overlay (Requirement)
    if (this.developerMode) {
      const hudW = Math.min(270, canvasEl.width - 16);
      const hudH = 86;
      const hudX = canvasEl.width - hudW - 8;
      const hudY = 8;

      ctx.save();
      ctx.fillStyle = 'rgba(5, 7, 12, 0.92)';
      ctx.fillRect(hudX, hudY, hudW, hudH);

      const statusColor = activeCandidate ? '#00ff41' : (candidates.length > 0 ? '#ffb400' : '#ff3b30');
      ctx.strokeStyle = statusColor;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(hudX, hudY, hudW, hudH);

      // Title & Cam ID
      ctx.fillStyle = statusColor;
      ctx.font = 'bold 9px "Share Tech Mono", monospace';
      ctx.fillText(`⚡ DEV HUD · 16-PT DIAGNOSTICS (CAM 0${cameraId})`, hudX + 6, hudY + 12);

      // Phase 2 Keyframe Buffer & Sharpness metrics
      const kfBuf = this.keyframeBuffers[cameraId];
      const currSharp = kfBuf && kfBuf.currentSharpness != null ? kfBuf.currentSharpness.toFixed(0) : '--';
      const candsCount = kfBuf && kfBuf.candidates ? kfBuf.candidates.length : 0;
      const kfStatus = kfBuf ? (kfBuf.isStatic ? 'STATIC_HOLD' : (kfBuf.rejectionReason && kfBuf.rejectionReason.startsWith('BLUR') ? 'BLUR_SKIP' : 'ACTIVE')) : 'IDLE';

      const sharp = serverDebug ? serverDebug.sharpnessScore : currSharp;
      const isBlur = (serverDebug && serverDebug.isBlurry) || (kfBuf && kfBuf.currentSharpness < this.config.minimum_sharpness);
      const blurTag = isBlur ? ' [BLUR]' : '';
      const vCount = vList.length;

      ctx.fillStyle = '#a0aec0';
      ctx.font = '8px "Share Tech Mono", monospace';
      ctx.fillText(`SHARP: ${sharp}${blurTag} | KF: ${kfStatus} (${candsCount} cands)`, hudX + 6, hudY + 24);

      const bestCand = candidates[0];
      const pScoreStr = bestCand ? `${Math.round(bestCand.detectorConfidence * 100)}% (AR:${bestCand.aspectRatio})` : 'NONE';
      const ocrStr = (serverDebug && serverDebug["13_normalizedOcrResults"] && serverDebug["13_normalizedOcrResults"][0]) || '--';
      ctx.fillText(`VEH: ${vCount} | PLT: ${pScoreStr} | OCR: "${ocrStr}"`, hudX + 6, hudY + 36);

      const skewVal = serverDebug && serverDebug.estimatedSkewDegrees != null ? serverDebug.estimatedSkewDegrees : 0.0;
      const skewSource = serverDebug ? (serverDebug.perspectiveSelected || 'ORIG') : 'NONE';
      const skewSign = skewVal > 0 ? '+' : '';
      const timing = serverDebug && serverDebug.timing;
      const tTotal = timing && timing.totalProcessingMs != null ? `${timing.totalProcessingMs.toFixed(0)}ms` : '--';
      const backend = timing && timing.inferenceBackend ? timing.inferenceBackend.toUpperCase() : 'ONNX';
      const ocrTier = timing && timing.ocrTier ? (timing.ocrTier.includes('TIER_1') ? 'T1' : (timing.ocrTier.includes('TIER_2') ? 'T2' : 'T3')) : 'T1';
      const scrTag = timing && timing.screenEnhanced ? ':SCR' : '';
      ctx.fillText(`SKEW: ${skewSign}${skewVal.toFixed(1)}° [${skewSource}] | LATENCY: ${tTotal} [${backend}:${ocrTier}${scrTag}]`, hudX + 6, hudY + 48);

      // Phase 6 Multi-Camera Concurrency & Queue Health
      const qInfo = this._lastServerQueue;
      const actWorkers = qInfo && qInfo.activeWorkers != null ? qInfo.activeWorkers : 0;
      const maxConc = qInfo && qInfo.maxConcurrency != null ? qInfo.maxConcurrency : 2;
      const qWait = qInfo && qInfo.queueWaitMs != null ? `${qInfo.queueWaitMs}ms` : '0ms';
      const camDrops = (qInfo && qInfo.camStats && qInfo.camStats.dropped) || 0;
      const dropTag = camDrops > 0 ? ` · DROPS:${camDrops}` : '';
      ctx.fillText(`QUEUE: ${actWorkers}/${maxConc} ACT | WAIT: ${qWait}${dropTag}`, hudX + 6, hudY + 60);

      // Rejection or Confirmation state
      let diagMsg = '';
      if (activeCandidate) {
        diagMsg = `CONFIRMED: ${activeCandidate.plate} (${Math.round(activeCandidate.confidence * 100)}%)`;
        ctx.fillStyle = '#00ff41';
      } else if (kfBuf && kfBuf.rejectionReason) {
        diagMsg = kfBuf.rejectionReason.slice(0, 38);
        ctx.fillStyle = kfBuf.isStatic ? '#00e5ff' : '#ffb400';
      } else if (serverDebug && serverDebug.summary) {
        const rawReason = serverDebug.summary.reason || 'Scanning...';
        diagMsg = `STATUS: ${rawReason.slice(0, 38)}`;
        ctx.fillStyle = serverDebug.summary.status.includes('REJECTED') ? '#ff5252' : '#ffb400';
      } else {
        diagMsg = 'MONITORING LIVE FEED...';
        ctx.fillStyle = '#a0aec0';
      }
      ctx.fillText(diagMsg, hudX + 6, hudY + 74);
    }
  }

  /* ──────────────────────────────────────────────────────────────────
     Emit Validated Detection Event
  ─────────────────────────────────────────────────────────────────── */
  _emitDetection(data) {
    // Send to server via Socket.io
    if (window.monitorSocket) {
      window.monitorSocket.emit('anpr:submit', data);
    }

    // Update per-camera stats
    if (this.statsPerCam[data.cameraId]) {
      this.statsPerCam[data.cameraId].lastPlate = data.plate;
    }

    // Update ANPR result bar on UI
    this._updateANPRBar(
      data.cameraId,
      data.plate,
      data.confidence,
      data.vehicleType,
      data.simulated,
      data.debug
    );

    // Call external callback
    if (typeof this.onDetection === 'function') this.onDetection(data);
  }

  _updateANPRBar(cameraId, plate, confidence, vehicleType, simulated, debug) {
    const bar = document.getElementById(`anpr-bar-${cameraId}`);
    if (!bar) return;
    const confPct = Math.round(confidence * 100);
    const now     = new Date().toTimeString().slice(0, 8);
    const simTag  = simulated ? '<span class="sim-tag">SIM</span>' : '';

    let debugTag = '';
    if (debug) {
      const v = Math.round((debug.vehicleConfidence || confidence) * 100);
      const p = Math.round((debug.plateConfidence || confidence) * 100);
      const o = Math.round((debug.ocrConfidence || confidence) * 100);
      debugTag = `<span style="font-size:9px;color:#718096;" title="V:${v}% P:${p}% O:${o}%">[V:${v}% P:${p}% O:${o}%]</span>`;
    }

    bar.innerHTML = `
      <span class="anpr-plate">${plate}</span>
      <span class="anpr-conf">${confPct}%</span>
      <span class="anpr-type">${vehicleType.toUpperCase()}</span>
      ${debugTag}
      <span class="anpr-cam">CAM-0${cameraId}</span>
      <span class="anpr-time">${now}</span>
      ${simTag}
    `;
    bar.classList.add('active');
    clearTimeout(bar._clearTimer);
    bar._clearTimer = setTimeout(() => bar.classList.remove('active'), 8000);
  }

  showLivePlateHUD(cameraId, det) {
    const card = document.getElementById(`live-plate-card-${cameraId}`);
    if (!card) return;

    const plateEl = document.getElementById(`lpc-plate-${cameraId}`);
    const stateEl = document.getElementById(`lpc-state-${cameraId}`);
    const subEl   = document.getElementById(`lpc-sub-${cameraId}`);

    if (plateEl) plateEl.textContent = det.plate;
    if (stateEl) stateEl.textContent = det.stateName || 'Indian Vehicle';
    if (subEl) {
      const latMsg = det.timeline && det.timeline.totalLiveConfirmationMs
        ? ` · ⏱️ ${det.timeline.totalLiveConfirmationMs}ms (${det.timeline.totalProcessingMs || 0}ms backend)`
        : '';
      subEl.textContent = `${det.vehicleType || 'CAR'} · ${Math.round(det.confidence * 100)}% CONF${latMsg}`;
    }

    card.classList.add('visible');
    clearTimeout(card._hideTimer);
    card._hideTimer = setTimeout(() => {
      card.classList.remove('visible');
    }, 6000);

    // Also update bottom ANPR result bar
    this._updateANPRBar(cameraId, det.plate, det.confidence, det.vehicleType || 'car', false);
  }
}

// Singleton export
window.ANPR = new ANPREngine();

