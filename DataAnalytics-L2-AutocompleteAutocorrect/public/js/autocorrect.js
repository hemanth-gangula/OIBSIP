/**
 * autocorrect.js — Autocorrect Playground frontend.
 *
 * Calls POST /api/autocorrect with the actual word from the input.
 * NEVER hardcodes corrections.
 */

(function () {
  'use strict';

  /* ——— DOM refs ——— */
  const wordInput           = document.getElementById('word-input');
  const checkBtn            = document.getElementById('check-btn');
  const clearBtn            = document.getElementById('clear-btn');
  const loadingArea         = document.getElementById('loading-area');
  const errorArea           = document.getElementById('error-area');
  const emptyState          = document.getElementById('empty-state');
  const resultsArea         = document.getElementById('results-area');
  const wordDisplay         = document.getElementById('word-display');
  const correctionsContainer = document.getElementById('corrections-container');

  /* ——— Get selected approach ——— */
  function getSelectedApproach() {
    const checked = document.querySelector('input[name="approach"]:checked');
    return checked ? checked.value : 'both';
  }

  /* ——— Build a correction item ——— */
  function buildCorrectionItem(item, approach) {
    const div = document.createElement('div');
    div.className = 'correction-item';

    const rankBadge = document.createElement('span');
    rankBadge.className = 'correction-item__rank';
    rankBadge.setAttribute('aria-hidden', 'true');
    rankBadge.textContent = item.rank;

    const wordSpan = document.createElement('span');
    wordSpan.className = 'correction-item__word';
    wordSpan.textContent = item.word;

    const scoreSpan = document.createElement('span');
    scoreSpan.className = 'correction-item__score';

    if (approach === 'edit_distance') {
      scoreSpan.textContent = 'dist: ' + item.edit_distance + '  freq: ' + (item.frequency || 0);
    } else {
      // probabilistic
      scoreSpan.textContent = 'p: ' + formatScore(item.probability, 6);
    }

    div.appendChild(rankBadge);
    div.appendChild(wordSpan);
    div.appendChild(scoreSpan);
    return div;
  }

  /* ——— Build one correction panel (title + list) ——— */
  function buildCorrectionPanel(title, badgeClass, items, approach, emptyMsg) {
    const panel = document.createElement('div');
    panel.className = 'correction-panel';

    const titleEl = document.createElement('h3');
    titleEl.className = 'correction-panel__title';

    const badge = document.createElement('span');
    badge.className = 'badge ' + badgeClass;
    badge.textContent = title;
    titleEl.appendChild(badge);
    panel.appendChild(titleEl);

    if (!items || items.length === 0) {
      const p = document.createElement('p');
      p.style.cssText = 'color:var(--muted);font-size:.875rem;padding:.25rem 0';
      p.textContent = emptyMsg || 'No suggestions found.';
      panel.appendChild(p);
    } else {
      items.forEach(function (item) {
        panel.appendChild(buildCorrectionItem(item, approach));
      });
    }

    return panel;
  }

  /* ——— Display results ——— */
  function displayResults(data) {
    if (!correctionsContainer || !wordDisplay) return;

    correctionsContainer.innerHTML = '';

    const approach = data.approach_used || 'both';

    // Word display line
    wordDisplay.innerHTML = '';
    const labelSpan = document.createElement('span');
    labelSpan.className = 'word-display__label';
    labelSpan.textContent = 'Input: ';
    const wordSpan = document.createElement('span');
    wordSpan.className = 'word-display__word';
    wordSpan.textContent = '"' + data.input_word + '"';
    wordDisplay.appendChild(labelSpan);
    wordDisplay.appendChild(wordSpan);

    if (data.is_correct) {
      const correctBadge = document.createElement('span');
      correctBadge.className = 'badge badge--green';
      correctBadge.textContent = '✓ Correctly Spelled';
      wordDisplay.appendChild(correctBadge);
    }

    // Corrections panels
    if (approach === 'both') {
      correctionsContainer.className = 'corrections-container corrections-container--side-by-side';

      correctionsContainer.appendChild(
        buildCorrectionPanel('Edit Distance', 'badge--blue', data.edit_distance, 'edit_distance',
          'No edit-distance candidates found.')
      );
      correctionsContainer.appendChild(
        buildCorrectionPanel('Probabilistic', 'badge--teal', data.probabilistic, 'probabilistic',
          'No probabilistic candidates found.')
      );
    } else if (approach === 'edit_distance') {
      correctionsContainer.className = 'corrections-container';
      correctionsContainer.appendChild(
        buildCorrectionPanel('Edit Distance Corrections', 'badge--blue', data.edit_distance, 'edit_distance',
          'No edit-distance candidates found.')
      );
    } else {
      correctionsContainer.className = 'corrections-container';
      correctionsContainer.appendChild(
        buildCorrectionPanel('Probabilistic Corrections', 'badge--teal', data.probabilistic, 'probabilistic',
          'No probabilistic candidates found.')
      );
    }

    if (emptyState)  emptyState.hidden  = true;
    if (resultsArea) resultsArea.hidden = false;
  }

  /* ——— Core check handler ——— */
  function handleCheck() {
    if (!wordInput) return;

    const word = wordInput.value.trim();

    // Reset UI
    clearError('error-area');
    if (resultsArea) resultsArea.hidden = true;
    if (emptyState)  emptyState.hidden  = true;

    // Validate: non-empty
    if (!word) {
      showError('error-area', 'Please enter a word to check.');
      if (emptyState) emptyState.hidden = false;
      return;
    }

    // Validate: single word (warn about multi-word; don't silently drop)
    if (/\s/.test(word)) {
      showError('error-area', 'Please enter a single word only (no spaces). Autocorrect works one word at a time.');
      if (emptyState) emptyState.hidden = false;
      return;
    }

    // Validate: letters only
    if (!/^[a-zA-Z]+$/.test(word)) {
      showError('error-area', 'Word must contain letters only (a-z). Numbers and punctuation are not supported.');
      if (emptyState) emptyState.hidden = false;
      return;
    }

    const approach = getSelectedApproach();

    // Show loading
    showLoading('loading-area');
    if (checkBtn) checkBtn.disabled = true;

    fetch('/api/autocorrect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word: word.toLowerCase(), approach: approach }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        });
      })
      .then(function (result) {
        hideLoading('loading-area');
        if (checkBtn) checkBtn.disabled = false;

        if (!result.ok) {
          const msg = (result.data && result.data.error) || ('Request failed with status ' + result.status);
          showError('error-area', msg);
          if (emptyState) emptyState.hidden = false;
          return;
        }

        displayResults(result.data);
      })
      .catch(function (err) {
        hideLoading('loading-area');
        if (checkBtn) checkBtn.disabled = false;
        showError('error-area', 'Network error — could not reach the server. ' + (err.message || ''));
        if (emptyState) emptyState.hidden = false;
      });
  }

  /* ——— Clear handler ——— */
  function handleClear() {
    if (wordInput) {
      wordInput.value = '';
      wordInput.focus();
    }
    clearError('error-area');
    if (resultsArea) resultsArea.hidden = true;
    if (emptyState)  emptyState.hidden  = false;
    hideLoading('loading-area');
  }

  /* ——— Event listeners ——— */
  if (checkBtn) {
    checkBtn.addEventListener('click', handleCheck);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', handleClear);
  }

  // Enter key on word input
  if (wordInput) {
    wordInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleCheck();
      }
    });
  }

  // Sample misspelled word buttons
  document.querySelectorAll('.btn--sample[data-word]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      if (wordInput) {
        wordInput.value = this.dataset.word;
        wordInput.focus();
        // Automatically check
        handleCheck();
      }
    });
  });

})();
