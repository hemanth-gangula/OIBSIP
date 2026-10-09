/**
 * autocomplete.js — Autocomplete Playground frontend.
 *
 * Calls POST /api/autocomplete with the actual text from the textarea.
 * NEVER hardcodes predictions.
 */

(function () {
  'use strict';

  /* ——— DOM refs ——— */
  const textarea    = document.getElementById('text-input');
  const submitBtn   = document.getElementById('submit-btn');
  const clearBtn    = document.getElementById('clear-btn');
  const charCount   = document.getElementById('char-count');
  const loadingArea = document.getElementById('loading-area');
  const errorArea   = document.getElementById('error-area');
  const emptyState  = document.getElementById('empty-state');
  const resultsArea = document.getElementById('results-area');
  const contextDisp = document.getElementById('context-display');
  const bigramDiv   = document.getElementById('bigram-results');
  const trigramDiv  = document.getElementById('trigram-results');
  const bigramList  = document.getElementById('bigram-list');
  const trigramList = document.getElementById('trigram-list');

  /* ——— Character counter ——— */
  if (textarea && charCount) {
    textarea.addEventListener('input', function () {
      charCount.textContent = this.value.length + ' / ' + this.maxLength;
    });
  }

  /* ——— Get selected model ——— */
  function getSelectedModel() {
    const checked = document.querySelector('input[name="model"]:checked');
    return checked ? checked.value : 'both';
  }

  /* ——— Build a suggestion card li ——— */
  function buildSuggestionCard(item, colorClass) {
    const li = document.createElement('li');
    li.className = 'suggestion-card';

    const rankBadge = document.createElement('span');
    rankBadge.className = 'suggestion-card__rank' + (colorClass ? ' ' + colorClass : '');
    rankBadge.setAttribute('aria-hidden', 'true');
    rankBadge.textContent = item.rank;

    const wordSpan = document.createElement('span');
    wordSpan.className = 'suggestion-card__word';
    wordSpan.textContent = item.word;

    const scoreSpan = document.createElement('span');
    scoreSpan.className = 'suggestion-card__score';
    scoreSpan.textContent = 'score: ' + formatScore(item.score, 4);

    li.appendChild(rankBadge);
    li.appendChild(wordSpan);
    li.appendChild(scoreSpan);
    return li;
  }

  /* ——— Display results ——— */
  function displayResults(data) {
    // Context words
    if (contextDisp) {
      if (data.context_words && data.context_words.length > 0) {
        contextDisp.textContent = 'Context used: ' + data.context_words.join(' → ');
        contextDisp.hidden = false;
      } else {
        contextDisp.hidden = true;
      }
    }

    const model = data.model_used || 'both';
    const showBigram  = model === 'bigram'  || model === 'both';
    const showTrigram = model === 'trigram' || model === 'both';

    // Bigram results
    if (bigramDiv && bigramList) {
      bigramList.innerHTML = '';
      if (showBigram && data.bigram && data.bigram.length > 0) {
        data.bigram.forEach(function (item) {
          bigramList.appendChild(buildSuggestionCard(item, ''));
        });
        bigramDiv.hidden = false;
      } else if (showBigram) {
        const li = document.createElement('li');
        li.style.cssText = 'padding:.5rem;color:var(--muted);font-size:.875rem;';
        li.textContent = 'No bigram suggestions found for this context.';
        bigramList.appendChild(li);
        bigramDiv.hidden = false;
      } else {
        bigramDiv.hidden = true;
      }
    }

    // Trigram results
    if (trigramDiv && trigramList) {
      trigramList.innerHTML = '';
      if (showTrigram && data.trigram && data.trigram.length > 0) {
        data.trigram.forEach(function (item) {
          trigramList.appendChild(buildSuggestionCard(item, 'suggestion-card__rank--teal'));
        });
        trigramDiv.hidden = false;
      } else if (showTrigram) {
        const li = document.createElement('li');
        li.style.cssText = 'padding:.5rem;color:var(--muted);font-size:.875rem;';
        li.textContent = 'No trigram suggestions found — trigrams need at least 2 words of context.';
        bigramList && (li.id = '');
        trigramList.appendChild(li);
        trigramDiv.hidden = false;
      } else {
        trigramDiv.hidden = true;
      }
    }

    // Show results panel
    if (emptyState) emptyState.hidden = true;
    if (resultsArea) resultsArea.hidden = false;
  }

  /* ——— Core submit handler ——— */
  function handleSubmit() {
    if (!textarea) return;

    const text = textarea.value.trim();

    // Reset UI
    clearError('error-area');
    if (resultsArea) resultsArea.hidden = true;
    if (emptyState)  emptyState.hidden  = true;

    // Validate
    if (!text) {
      showError('error-area', 'Please enter some text before requesting suggestions.');
      if (emptyState) emptyState.hidden = false;
      return;
    }

    const model = getSelectedModel();

    // Show loading
    showLoading('loading-area');
    if (submitBtn) submitBtn.disabled = true;

    fetch('/api/autocomplete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, model: model, top_k: 3 }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        });
      })
      .then(function (result) {
        hideLoading('loading-area');
        if (submitBtn) submitBtn.disabled = false;

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
        if (submitBtn) submitBtn.disabled = false;
        showError('error-area', 'Network error — could not reach the server. ' + (err.message || ''));
        if (emptyState) emptyState.hidden = false;
      });
  }

  /* ——— Clear handler ——— */
  function handleClear() {
    if (textarea) {
      textarea.value = '';
      if (charCount) charCount.textContent = '0 / ' + (textarea.maxLength || 500);
      textarea.focus();
    }
    clearError('error-area');
    if (resultsArea) resultsArea.hidden = true;
    if (bigramDiv)   bigramDiv.hidden   = true;
    if (trigramDiv)  trigramDiv.hidden  = true;
    if (emptyState)  emptyState.hidden  = false;
    hideLoading('loading-area');
  }

  /* ——— Event listeners ——— */
  if (submitBtn) {
    submitBtn.addEventListener('click', handleSubmit);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', handleClear);
  }

  // Ctrl+Enter on textarea
  if (textarea) {
    textarea.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSubmit();
      }
    });
  }

  // Sample context buttons
  document.querySelectorAll('.btn--sample[data-sample]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      if (textarea) {
        textarea.value = this.dataset.sample;
        if (charCount) charCount.textContent = textarea.value.length + ' / ' + (textarea.maxLength || 500);
        textarea.focus();
        // Automatically submit
        handleSubmit();
      }
    });
  });

})();
