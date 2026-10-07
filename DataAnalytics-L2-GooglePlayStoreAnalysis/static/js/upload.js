/* ================================================================
   upload.js – handles file upload, preview, and analysis trigger
   ================================================================ */

const APPS_REQUIRED    = ['App','Category','Rating','Reviews','Size','Installs','Type','Price'];
const REVIEWS_REQUIRED = ['App','Translated_Review','Sentiment','Sentiment_Polarity','Sentiment_Subjectivity'];

let appsUploaded    = false;
let reviewsUploaded = false;

/* ── Drag-and-drop wiring ───────────────────── */
['dz-apps','dz-reviews'].forEach(id => {
  const dz   = document.getElementById(id);
  const inp  = dz.querySelector('input[type="file"]');
  const kind = id === 'dz-apps' ? 'apps' : 'reviews';

  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, kind);
  });

  inp.addEventListener('change', () => {
    if (inp.files[0]) handleFile(inp.files[0], kind);
  });
});

/* ── Handle a chosen file ───────────────────── */
function handleFile(file, kind) {
  setStatus(kind, 'loading', `<span class="spinner"></span> Uploading ${file.name}…`);

  const fd = new FormData();
  fd.append('file', file);

  fetch(`/upload/${kind}`, { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (!data.success) {
        setStatus(kind, 'error', `⚠️ ${data.error}`);
        return;
      }
      setStatus(kind, 'success', `✅ ${data.message}`);
      renderFileInfo(kind, data);
      if (kind === 'apps')    { appsUploaded    = true; markCheck('chk-apps');    }
      if (kind === 'reviews') { reviewsUploaded = true; markCheck('chk-reviews'); }
      updateAnalyseBtn();
    })
    .catch(err => setStatus(kind, 'error', `⚠️ Network error: ${err}`));
}

/* ── Render file info panel ─────────────────── */
function renderFileInfo(kind, data) {
  const required = kind === 'apps' ? APPS_REQUIRED : REVIEWS_REQUIRED;

  // Meta row
  const metaEl = document.getElementById(`meta-${kind}`);
  metaEl.innerHTML = [
    { label: 'File',    val: data.filename },
    { label: 'Format',  val: data.filetype },
    { label: 'Rows',    val: Number(data.rows).toLocaleString() },
    { label: 'Columns', val: data.cols },
  ].map(m => `
    <div class="file-meta-item">
      <span>${m.label}</span>
      <strong>${m.val}</strong>
    </div>`).join('');

  // Column tags
  const colsEl = document.getElementById(`cols-${kind}`);
  colsEl.innerHTML = data.columns.map(c => {
    const isRequired = required.includes(c);
    return `<span class="col-tag ${isRequired ? 'required' : ''}" 
                  title="${isRequired ? 'Required column ✓' : ''}">${c}</span>`;
  }).join('');

  // Preview table
  const tbl = document.getElementById(`tbl-${kind}`);
  if (data.preview && data.preview.length) {
    const cols = Object.keys(data.preview[0]);
    tbl.innerHTML = `
      <thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
      <tbody>${data.preview.map(row =>
        `<tr>${cols.map(c => `<td>${row[c] ?? ''}</td>`).join('')}</tr>`
      ).join('')}</tbody>`;
  }

  document.getElementById(`info-${kind}`).classList.add('show');
}

/* ── Status helper ──────────────────────────── */
function setStatus(kind, type, html) {
  const el = document.getElementById(`status-${kind}`);
  el.innerHTML = `<div class="status-msg ${type}">${html}</div>`;
}

/* ── Checklist helper ───────────────────────── */
function markCheck(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

/* ── Enable/disable analyse button ─────────── */
function updateAnalyseBtn() {
  document.getElementById('btn-analyse').disabled = !(appsUploaded && reviewsUploaded);
}

/* ── Run analysis ───────────────────────────── */
const STEPS = [
  { id: 'chk-clean',   label: 'Cleaning datasets…',               pct: 15 },
  { id: 'chk-cat',     label: 'Analysing categories…',            pct: 28 },
  { id: 'chk-rating',  label: 'Analysing ratings…',               pct: 42 },
  { id: 'chk-size',    label: 'Analysing size & installs…',       pct: 55 },
  { id: 'chk-price',   label: 'Analysing pricing…',               pct: 65 },
  { id: 'chk-sent',    label: 'Running sentiment analysis…',      pct: 78 },
  { id: 'chk-sentcat', label: 'Merging sentiment by category…',   pct: 87 },
  { id: 'chk-plotly',  label: 'Generating interactive charts…',   pct: 94 },
  { id: 'chk-insights','label': 'Building developer insights…',   pct: 98 },
];

function runAnalysis() {
  const btn = document.getElementById('btn-analyse');
  btn.disabled = true;

  document.getElementById('progress-wrap').classList.add('show');
  document.getElementById('analyse-error').innerHTML = '';

  let step = 0;
  const timer = setInterval(() => {
    if (step >= STEPS.length) { clearInterval(timer); return; }
    const s = STEPS[step++];
    markCheck(s.id);
    setProgress(s.pct, s.label);
  }, 600);

  fetch('/analyse', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      clearInterval(timer);
      if (!data.success) {
        setProgress(0, '');
        document.getElementById('progress-wrap').classList.remove('show');
        document.getElementById('analyse-error').innerHTML =
          `<div class="status-msg error mt-2">⚠️ ${data.error}</div>`;
        btn.disabled = false;
        return;
      }
      STEPS.forEach(s => markCheck(s.id));
      setProgress(100, '✅ Analysis complete! Redirecting to dashboard…');
      setTimeout(() => { window.location.href = '/dashboard'; }, 900);
    })
    .catch(err => {
      clearInterval(timer);
      document.getElementById('analyse-error').innerHTML =
        `<div class="status-msg error mt-2">⚠️ Network error: ${err}</div>`;
      btn.disabled = false;
    });
}

function setProgress(pct, label) {
  document.getElementById('progress-fill').style.width  = pct + '%';
  document.getElementById('progress-label').textContent = label;
}

/* ── Poll status on page load (restore state) ─ */
fetch('/status')
  .then(r => r.json())
  .then(s => {
    if (s.apps_uploaded)    { appsUploaded    = true; markCheck('chk-apps'); }
    if (s.reviews_uploaded) { reviewsUploaded = true; markCheck('chk-reviews'); }
    updateAnalyseBtn();
  });
