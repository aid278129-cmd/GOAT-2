'use strict';

/**
 * Geographic Calculation Utilities
 * Multi-Camera Indian ANPR Traffic Intelligence Platform
 */

const EARTH_RADIUS_KM = 6371.0;

/**
 * Validates whether a value is a finite number within the given range.
 */
function isValidCoord(val, min, max) {
  if (val === null || val === undefined || val === '') return false;
  const num = Number(val);
  return !isNaN(num) && isFinite(num) && num >= min && num <= max;
}

/**
 * Calculates geographic distance between two coordinates in kilometers using the Haversine formula.
 *
 * @param {number} lat1 Latitude of point 1 (-90 to 90)
 * @param {number} lon1 Longitude of point 1 (-180 to 180)
 * @param {number} lat2 Latitude of point 2 (-90 to 90)
 * @param {number} lon2 Longitude of point 2 (-180 to 180)
 * @returns {number|null} Distance in kilometers or null if invalid inputs
 */
function haversineDistance(lat1, lon1, lat2, lon2) {
  // Validate all coordinates strictly
  if (
    !isValidCoord(lat1, -90, 90) ||
    !isValidCoord(lon1, -180, 180) ||
    !isValidCoord(lat2, -90, 90) ||
    !isValidCoord(lon2, -180, 180)
  ) {
    return null;
  }

  const nLat1 = Number(lat1);
  const nLon1 = Number(lon1);
  const nLat2 = Number(lat2);
  const nLon2 = Number(lon2);

  // Exact same point check
  if (nLat1 === nLat2 && nLon1 === nLon2) {
    return 0;
  }

  const toRad = (angle) => (angle * Math.PI) / 180;

  const dLat = toRad(nLat2 - nLat1);
  const dLon = toRad(nLon2 - nLon1);

  const radLat1 = toRad(nLat1);
  const radLat2 = toRad(nLat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(radLat1) * Math.cos(radLat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);

  const clampedA = Math.min(1, Math.max(0, a));
  const c = 2 * Math.atan2(Math.sqrt(clampedA), Math.sqrt(1 - clampedA));

  const distance = EARTH_RADIUS_KM * c;
  return isNaN(distance) || !isFinite(distance) ? null : distance;
}

/**
 * Calculates estimated average speed across a travel segment.
 *
 * @param {number} distanceKm Distance in kilometers
 * @param {number} travelTimeSeconds Travel time in seconds
 * @returns {number|null} Estimated average speed in km/h or null if invalid
 */
function calculateEstimatedSpeed(distanceKm, travelTimeSeconds) {
  if (
    distanceKm === null ||
    distanceKm === undefined ||
    isNaN(distanceKm) ||
    !isFinite(distanceKm) ||
    distanceKm < 0
  ) {
    return null;
  }

  if (
    travelTimeSeconds === null ||
    travelTimeSeconds === undefined ||
    isNaN(travelTimeSeconds) ||
    !isFinite(travelTimeSeconds) ||
    travelTimeSeconds <= 0
  ) {
    return null;
  }

  const travelTimeHours = travelTimeSeconds / 3600;
  const speedKmh = distanceKm / travelTimeHours;

  if (isNaN(speedKmh) || !isFinite(speedKmh)) {
    return null;
  }

  return speedKmh;
}

module.exports = {
  haversineDistance,
  calculateEstimatedSpeed,
};
