/**
 * main.js — Shared utility functions and nav toggle.
 * Loaded on every page via base.html.
 */

/* =====================================================
   Utility: show / hide a loading spinner
   ===================================================== */

/**
 * Show a loading spinner inside the container identified by `containerId`.
 * Hides the container first, then unhides the sibling loading element.
 * @param {string} loadingId - id of the loading element
 */
function showLoading(loadingId) {
  const el = document.getElementById(loadingId);
  if (el) {
    el.hidden = false;
  }
}

/**
 * Hide the loading spinner.
 * @param {string} loadingId - id of the loading element
 */
function hideLoading(loadingId) {
  const el = document.getElementById(loadingId);
  if (el) {
    el.hidden = true;
  }
}

/* =====================================================
   Utility: error messages
   ===================================================== */

/**
 * Display an error message in the given element.
 * @param {string} errorId - id of the error container element
 * @param {string} message - human-readable error text
 */
function showError(errorId, message) {
  const el = document.getElementById(errorId);
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
}

/**
 * Clear an error message area.
 * @param {string} errorId - id of the error container element
 */
function clearError(errorId) {
  const el = document.getElementById(errorId);
  if (!el) return;
  el.textContent = '';
  el.hidden = true;
}

/* =====================================================
   Utility: score formatting
   ===================================================== */

/**
 * Format a numeric score to a fixed number of decimal places.
 * Returns '—' for null/undefined/NaN.
 * @param {number|null|undefined} score
 * @param {number} [decimals=4]
 * @returns {string}
 */
function formatScore(score, decimals = 4) {
  if (score === null || score === undefined || isNaN(score)) return '—';
  return Number(score).toFixed(decimals);
}

/* =====================================================
   Utility: debounce
   ===================================================== */

/**
 * Returns a debounced version of `fn` that delays invocation by `delay` ms.
 * @param {Function} fn
 * @param {number} delay - milliseconds
 * @returns {Function}
 */
function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

/* =====================================================
   Mobile nav toggle
   ===================================================== */

(function initNav() {
  const hamburger = document.getElementById('nav-hamburger');
  const mobileMenu = document.getElementById('mobile-menu');

  if (!hamburger || !mobileMenu) return;

  hamburger.addEventListener('click', function () {
    const isOpen = this.getAttribute('aria-expanded') === 'true';
    const nextState = !isOpen;

    this.setAttribute('aria-expanded', String(nextState));
    mobileMenu.setAttribute('aria-hidden', String(!nextState));
    mobileMenu.classList.toggle('is-open', nextState);
  });

  // Close menu when clicking a nav link inside it
  mobileMenu.querySelectorAll('.nav__mobile-link').forEach(function (link) {
    link.addEventListener('click', function () {
      hamburger.setAttribute('aria-expanded', 'false');
      mobileMenu.setAttribute('aria-hidden', 'true');
      mobileMenu.classList.remove('is-open');
    });
  });

  // Close menu on Escape key
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && hamburger.getAttribute('aria-expanded') === 'true') {
      hamburger.setAttribute('aria-expanded', 'false');
      mobileMenu.setAttribute('aria-hidden', 'true');
      mobileMenu.classList.remove('is-open');
      hamburger.focus();
    }
  });
})();
