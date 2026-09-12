/**
 * camera-config.js
 * Central camera node configuration.
 * Add more cameras here — the rest of the system adapts automatically.
 */
const CAMERA_NODES = [
  {
    id: 1,
    name: 'Junction A',
    location: 'Chennai Central',
    zone: 'North',
    lat: 13.0827,
    lng: 80.2707,
  },
  {
    id: 2,
    name: 'Junction B',
    location: 'T. Nagar',
    zone: 'West',
    lat: 13.0731,
    lng: 80.2609,
  },
  {
    id: 3,
    name: 'Junction C',
    location: 'Perambur',
    zone: 'North-East',
    lat: 13.0878,
    lng: 80.2785,
  },
  {
    id: 4,
    name: 'Highway Entry',
    location: 'Guindy',
    zone: 'South',
    lat: 13.0569,
    lng: 80.2425,
  },
];

function getCameraNode(id) {
  return CAMERA_NODES.find(c => c.id === parseInt(id, 10)) || null;
}
