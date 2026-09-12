/**
 * shared.js — Common utilities and navigation for all pages
 */

/* ── Navigation injection ── */
function injectNav(activePage) {
  const pages = [
    { href: '/dashboard.html',  icon: '🎥', label: 'Command Center',   id: 'cmd'  },
    { href: '/tracking.html',   icon: '🚗', label: 'Vehicle Tracking', id: 'track'},
    { href: '/analytics.html',  icon: '📊', label: 'Analytics',        id: 'stats'},
    { href: '/network.html',    icon: '📡', label: 'Camera Network',   id: 'net'  },
    { href: '/watchlist.html',  icon: '⚠️',  label: 'Watchlist',        id: 'wl'   },
  ];

  const nav = document.createElement('nav');
  nav.className = 'top-nav';
  nav.innerHTML = `
    <div class="nav-brand">
      <div class="nav-logo"></div>
      <span class="nav-title">ANPR Intelligence</span>
    </div>
    <div class="nav-links">
      ${pages.map(p => `
        <a href="${p.href}" class="nav-link ${p.id === activePage ? 'active' : ''}" id="navlink-${p.id}">
          <span class="nav-icon">${p.icon}</span>
          <span class="nav-label">${p.label}</span>
        </a>
      `).join('')}
    </div>
    <div class="nav-meta">
      <div class="nav-alert-badge hidden" id="navAlertBadge">
        <span id="navAlertCount">0</span> ALERT
      </div>
      <span class="nav-clock" id="navClock">--:--:--</span>
    </div>
  `;
  document.body.insertBefore(nav, document.body.firstChild);

  // Clock
  function tickNav() {
    const el = document.getElementById('navClock');
    if (el) el.textContent = new Date().toTimeString().slice(0, 8);
  }
  setInterval(tickNav, 1000);
  tickNav();
}

/* ── Format helpers ── */
function formatPlate(plate) {
  return plate ? plate.toUpperCase() : '—';
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toTimeString().slice(0, 8);
}

function formatDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('en-IN', { hour12: false, dateStyle: 'short', timeStyle: 'medium' });
}

function formatConfidence(conf) {
  return `${Math.round(parseFloat(conf) * 100)}%`;
}

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60000)  return `${Math.round(diff / 1000)}s ago`;
  if (diff < 3600000) return `${Math.round(diff / 60000)}m ago`;
  return `${Math.round(diff / 3600000)}h ago`;
}

/* ── API helpers ── */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ── Alert badge ── */
let _alertCount = 0;
function incrementAlertBadge() {
  _alertCount++;
  const badge = document.getElementById('navAlertBadge');
  const cnt   = document.getElementById('navAlertCount');
  if (badge) badge.classList.remove('hidden');
  if (cnt)   cnt.textContent = _alertCount;
}
