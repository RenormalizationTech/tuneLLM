/* Recorded static JSON viewer only. No inference endpoint or third-party dependency. */
(() => {
  'use strict';
  async function enhance(root) {
    const interactive = root.querySelector('[data-demo-interactive]');
    const fallback = root.querySelector('[data-demo-fallback]');
    if (!interactive || !fallback) return;
    try {
      const url = new URL(root.dataset.demoUrl, document.baseURI);
      if (url.origin !== window.location.origin) throw new Error('Static demo data must be same-origin');
      const response = await fetch(url.href, { credentials: 'same-origin', cache: 'force-cache' });
      if (!response.ok) throw new Error('Recorded data unavailable');
      const data = await response.json();
      const records = data.sarcasm && data.sarcasm.strengths;
      if (data.mode !== 'RECORDED' || !Array.isArray(records) || records.length !== 5 ||
          records.map(r => r.strength).join(',') !== '0,2,4,8,16' ||
          records.some(r => typeof r.output !== 'string' ||
            ['sarcasm', 'coherence', 'impoliteness'].some(k => !Number.isFinite(r.scores[k])))) {
        throw new Error('Recorded schema mismatch');
      }
      const heading = root.querySelector('[data-demo-heading]');
      const output = root.querySelector('[data-demo-output]');
      const cap = root.querySelector('[data-demo-cap]');
      const status = root.querySelector('[data-demo-status]');
      function select(strength) {
        const r = records.find(row => row.strength === strength);
        if (!r) return;
        heading.textContent = strength === 0 ? 'Strength 0 · ordinary generation' :
          `Strength ${strength}${strength === 16 ? ' · extreme' : ''}`;
        output.textContent = r.output; // Model text is data; never render it as HTML.
        for (const key of ['sarcasm', 'coherence', 'impoliteness']) {
          root.querySelector(`[data-demo-score="${key}"]`).textContent = String(r.scores[key]);
        }
        cap.textContent = `Recorded ${r.generated_tokens}-token answer · ${r.cap_hit ?
          'reached the cap; incomplete ending retained.' : 'original ending retained.'}`;
        status.textContent = `Reading saved strength ${strength}. Sarcasm ${r.scores.sarcasm}, coherence ${r.scores.coherence}, impoliteness ${r.scores.impoliteness}, each out of 100. No inference ran.`;
      }
      root.querySelectorAll('input[name="recorded-dose"]').forEach(input => {
        input.addEventListener('change', () => { if (input.checked) select(Number(input.value)); });
      });
      select(0);
      interactive.hidden = false;
      fallback.open = false; // Always available; without JS it remains open in the HTML.
      root.dataset.demoState = 'recorded-ready';
    } catch (error) {
      root.dataset.demoState = 'text-fallback';
      fallback.open = true;
      // The page already contains the complete, useful recorded text.
      console.info('Recorded viewer unavailable; static examples remain visible.');
    }
  }
  function init() { document.querySelectorAll('[data-recorded-sarcasm]').forEach(enhance); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
