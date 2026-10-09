/**
 * analytics.js — Analytics Dashboard frontend.
 *
 * Fetches /api/analytics and /api/metrics in parallel on DOMContentLoaded.
 * Renders Plotly charts from the real API data.
 */

(function () {
  'use strict';

  /* ——— Plotly layout defaults ——— */
  const PLOTLY_CONFIG = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'toggleSpikelines'],
  };

  const NAVY   = '#0f172a';
  const BLUE   = '#3b82f6';
  const TEAL   = '#14b8a6';
  const MUTED  = '#64748b';
  const BG     = '#ffffff';
  const BORDER = '#e2e8f0';

  const BASE_LAYOUT = {
    paper_bgcolor: BG,
    plot_bgcolor:  BG,
    font: { family: 'Inter, system-ui, sans-serif', color: NAVY, size: 12 },
    margin: { t: 20, r: 20, b: 60, l: 140 },
  };

  /* ——— DOM refs ——— */
  const statsLoading   = document.getElementById('stats-loading');
  const statsError     = document.getElementById('stats-error');
  const statsGrid      = document.getElementById('stats-grid');

  const chartLoadingTop = document.getElementById('chart-loading-top');
  const chartLoadingAc  = document.getElementById('chart-loading-ac');
  const chartLoadingAcr = document.getElementById('chart-loading-acr');

  /* ——— Helpers ——— */
  function fmt(n) {
    if (n === undefined || n === null) return '—';
    return Number(n).toLocaleString();
  }

  function setStatValue(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function hideElement(el) {
    if (el) el.hidden = true;
  }

  function showElement(el) {
    if (el) el.hidden = false;
  }

  function showChartError(containerId, message) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '<p style="color:var(--muted);font-size:.875rem;padding:1rem;">' +
      '⚠ ' + message + '</p>';
  }

  /* ——— Render top-20 words chart ——— */
  function renderTopWordsChart(topWords) {
    hideElement(chartLoadingTop);

    if (!topWords || topWords.length === 0) {
      showChartError('chart-top-words', 'No vocabulary data available.');
      return;
    }

    // Horizontal bar — words on y-axis, frequency on x-axis
    const words = topWords.map(function (d) { return d.word; });
    const freqs  = topWords.map(function (d) { return d.freq; });

    const trace = {
      type: 'bar',
      orientation: 'h',
      x: freqs,
      y: words,
      marker: {
        color: freqs.map(function (_, i) {
          // Gradient from teal (rank 1) to blue (rank 20)
          return i < 5 ? TEAL : BLUE;
        }),
        line: { color: BORDER, width: 0.5 },
      },
      hovertemplate: '<b>%{y}</b><br>Frequency: %{x:,}<extra></extra>',
    };

    const layout = Object.assign({}, BASE_LAYOUT, {
      margin: { t: 10, r: 30, b: 40, l: 90 },
      xaxis: {
        title: { text: 'Frequency', font: { size: 11 } },
        gridcolor: BORDER,
        showgrid: true,
      },
      yaxis: {
        autorange: 'reversed',
        tickfont: { size: 11 },
        gridcolor: BORDER,
      },
      height: 420,
    });

    Plotly.newPlot('chart-top-words', [trace], layout, PLOTLY_CONFIG);
  }

  /* ——— Render autocomplete comparison chart ——— */
  function renderAutocompleteChart(metrics) {
    hideElement(chartLoadingAc);

    if (!metrics || !metrics.autocomplete) {
      showChartError('chart-autocomplete-comparison', 'Metrics data unavailable.');
      return;
    }

    const ac = metrics.autocomplete;
    const categories = ['Top-1 Accuracy', 'Top-3 Accuracy', 'MRR'];

    const bigramValues  = [
      ac.bigram.top1,
      ac.bigram.top3,
      ac.bigram.mrr,
    ];
    const trigramValues = [
      ac.trigram.top1,
      ac.trigram.top3,
      ac.trigram.mrr,
    ];

    const traceBigram = {
      name: 'Bigram',
      type: 'bar',
      x: categories,
      y: bigramValues,
      marker: { color: BLUE, opacity: 0.85 },
      hovertemplate: 'Bigram<br>%{x}: %{y:.1%}<extra></extra>',
    };

    const traceTrigram = {
      name: 'Trigram',
      type: 'bar',
      x: categories,
      y: trigramValues,
      marker: { color: TEAL, opacity: 0.85 },
      hovertemplate: 'Trigram<br>%{x}: %{y:.1%}<extra></extra>',
    };

    const layout = Object.assign({}, BASE_LAYOUT, {
      barmode: 'group',
      margin: { t: 10, r: 20, b: 60, l: 60 },
      yaxis: {
        title: { text: 'Score', font: { size: 11 } },
        tickformat: '.0%',
        range: [0, 1],
        gridcolor: BORDER,
      },
      xaxis: { gridcolor: BORDER },
      legend: { orientation: 'h', y: -0.18 },
      height: 300,
    });

    Plotly.newPlot('chart-autocomplete-comparison', [traceBigram, traceTrigram], layout, PLOTLY_CONFIG);
  }

  /* ——— Render autocorrect comparison chart ——— */
  function renderAutocorrectChart(metrics) {
    hideElement(chartLoadingAcr);

    if (!metrics || !metrics.autocorrect) {
      showChartError('chart-autocorrect-comparison', 'Metrics data unavailable.');
      return;
    }

    const acr = metrics.autocorrect;
    const categories = ['Top-1 Accuracy', 'Top-3 Accuracy'];

    const edValues   = [acr.edit_distance.top1, acr.edit_distance.top3];
    const probValues = [acr.probabilistic.top1,  acr.probabilistic.top3];

    const traceEd = {
      name: 'Edit Distance',
      type: 'bar',
      x: categories,
      y: edValues,
      marker: { color: BLUE, opacity: 0.85 },
      hovertemplate: 'Edit Distance<br>%{x}: %{y:.1%}<extra></extra>',
    };

    const traceProb = {
      name: 'Probabilistic',
      type: 'bar',
      x: categories,
      y: probValues,
      marker: { color: TEAL, opacity: 0.85 },
      hovertemplate: 'Probabilistic<br>%{x}: %{y:.1%}<extra></extra>',
    };

    const layout = Object.assign({}, BASE_LAYOUT, {
      barmode: 'group',
      margin: { t: 10, r: 20, b: 60, l: 60 },
      yaxis: {
        title: { text: 'Score', font: { size: 11 } },
        tickformat: '.0%',
        range: [0, 1],
        gridcolor: BORDER,
      },
      xaxis: { gridcolor: BORDER },
      legend: { orientation: 'h', y: -0.18 },
      height: 300,
    });

    Plotly.newPlot('chart-autocorrect-comparison', [traceEd, traceProb], layout, PLOTLY_CONFIG);
  }

  /* ——— Render stats card ——— */
  function renderStats(analyticsData) {
    hideElement(statsLoading);

    const vs = analyticsData.vocab_stats || {};
    const top = (analyticsData.top_20_words && analyticsData.top_20_words[0]) || {};

    setStatValue('stat-vocab-size',    fmt(vs.unique_words));
    setStatValue('stat-total-tokens',  fmt(vs.total_words));
    setStatValue('stat-avg-freq',      vs.avg_freq !== undefined ? vs.avg_freq.toFixed(1) : '—');
    setStatValue('stat-top-word',      top.word ? '"' + top.word + '" (' + fmt(top.freq) + ')' : '—');

    showElement(statsGrid);
  }

  /* ——— Main: fetch analytics + metrics in parallel ——— */
  document.addEventListener('DOMContentLoaded', function () {
    var analyticsPromise = fetch('/api/analytics').then(function (r) { return r.json(); });
    var metricsPromise   = fetch('/api/metrics').then(function (r) { return r.json(); });

    // Stats + top-words chart from analytics
    analyticsPromise
      .then(function (data) {
        renderStats(data);
        // Wait for Plotly to be available (it's loaded with defer)
        waitForPlotly(function () {
          renderTopWordsChart(data.top_20_words);
        });
      })
      .catch(function (err) {
        hideElement(statsLoading);
        if (statsError) {
          statsError.textContent = 'Failed to load analytics data: ' + (err.message || 'unknown error');
          statsError.hidden = false;
        }
        hideElement(chartLoadingTop);
        showChartError('chart-top-words', 'Could not load corpus data.');
      });

    // Model comparison charts from metrics
    metricsPromise
      .then(function (data) {
        waitForPlotly(function () {
          renderAutocompleteChart(data);
          renderAutocorrectChart(data);
        });
      })
      .catch(function (err) {
        hideElement(chartLoadingAc);
        hideElement(chartLoadingAcr);
        showChartError('chart-autocomplete-comparison', 'Could not load metrics: ' + (err.message || ''));
        showChartError('chart-autocorrect-comparison',  'Could not load metrics: ' + (err.message || ''));
      });
  });

  /**
   * Wait until window.Plotly is available (it's loaded with defer), then call fn.
   * Retries every 50ms up to 5s.
   */
  function waitForPlotly(fn) {
    if (window.Plotly) {
      fn();
    } else {
      var attempts = 0;
      var interval = setInterval(function () {
        attempts++;
        if (window.Plotly) {
          clearInterval(interval);
          fn();
        } else if (attempts > 100) {
          clearInterval(interval);
          console.warn('Plotly did not load within 5 seconds.');
        }
      }, 50);
    }
  }

})();
