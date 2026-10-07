/**
 * upload.js
 * Handles drag-and-drop, file preview, form validation,
 * and loading-state animation for both upload forms.
 */

(function () {
  'use strict';

  // ── Utility ──────────────────────────────────────────────────────────────

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // ── Wire up one upload block ──────────────────────────────────────────────

  function initUpload(prefix) {
    var fileInput  = document.getElementById(prefix + '_file');
    var dropZone   = document.getElementById('drop-' + prefix);
    var dropText   = document.getElementById('drop-' + prefix + '-text');
    var preview    = document.getElementById('preview-' + prefix);
    var previewName= document.getElementById('preview-' + prefix + '-name');
    var submitBtn  = document.getElementById('btn-' + prefix);
    var form       = document.getElementById('form-' + prefix);

    if (!fileInput || !dropZone || !submitBtn || !form) return;

    // ── File selected via input ─────────────────────────────────────────────
    fileInput.addEventListener('change', function () {
      handleFile(fileInput.files[0]);
    });

    // ── Drag & drop ─────────────────────────────────────────────────────────
    dropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', function () {
      dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      var file = e.dataTransfer.files[0];
      if (file) {
        // Assign to the real input so FormData picks it up
        var dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        handleFile(file);
      }
    });

    // ── Handle file selection ────────────────────────────────────────────────
    function handleFile(file) {
      if (!file) return;

      var ext = file.name.split('.').pop().toLowerCase();
      var allowed = ['csv', 'xlsx', 'xls'];
      if (!allowed.includes(ext)) {
        showInlineError(dropZone, 'Only CSV or Excel files are accepted.');
        return;
      }

      var maxBytes = 50 * 1024 * 1024;
      if (file.size > maxBytes) {
        showInlineError(dropZone, 'File exceeds the 50 MB limit (' + formatBytes(file.size) + ').');
        return;
      }

      clearInlineError(dropZone);

      // Show preview
      previewName.textContent = file.name + '  (' + formatBytes(file.size) + ')';
      preview.hidden = false;
      dropText.innerHTML = 'File selected — <span class="drop-zone__link">change</span>';

      // Enable submit
      submitBtn.disabled = false;
    }

    // ── Form submit: show spinner ────────────────────────────────────────────
    form.addEventListener('submit', function (e) {
      if (submitBtn.disabled) { e.preventDefault(); return; }
      var textSpan    = submitBtn.querySelector('.btn-text');
      var spinnerSpan = submitBtn.querySelector('.btn-spinner');
      if (textSpan)    textSpan.hidden    = true;
      if (spinnerSpan) spinnerSpan.hidden = false;
      submitBtn.disabled = true;
    });

    // ── Inline error helpers ─────────────────────────────────────────────────
    function showInlineError(zone, msg) {
      clearInlineError(zone);
      var err = document.createElement('p');
      err.className = 'drop-zone__error';
      err.style.cssText = 'color:#fca5a5;font-size:.8rem;margin-top:6px;';
      err.textContent = '⚠ ' + msg;
      zone.parentNode.insertBefore(err, zone.nextSibling);
    }
    function clearInlineError(zone) {
      var existing = zone.parentNode.querySelector('.drop-zone__error');
      if (existing) existing.remove();
    }
  }

  // ── Clear file selection ──────────────────────────────────────────────────

  window.clearFile = function (prefix) {
    var fileInput   = document.getElementById(prefix + '_file');
    var preview     = document.getElementById('preview-' + prefix);
    var previewName = document.getElementById('preview-' + prefix + '-name');
    var submitBtn   = document.getElementById('btn-' + prefix);
    var dropText    = document.getElementById('drop-' + prefix + '-text');

    if (fileInput) fileInput.value = '';
    if (preview)   preview.hidden = true;
    if (previewName) previewName.textContent = '';
    if (submitBtn) submitBtn.disabled = true;
    if (dropText)  dropText.innerHTML = 'Drag &amp; drop or <span class="drop-zone__link">browse</span>';
  };

  // ── Auto-dismiss flash messages ───────────────────────────────────────────

  function autoDismissFlash() {
    var flashes = document.querySelectorAll('.flash');
    flashes.forEach(function (el) {
      setTimeout(function () {
        el.style.transition = 'opacity .4s';
        el.style.opacity = '0';
        setTimeout(function () { el.remove(); }, 450);
      }, 6000);
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────────

  document.addEventListener('DOMContentLoaded', function () {
    initUpload('apps');
    initUpload('reviews');
    autoDismissFlash();
  });

})();
