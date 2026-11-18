/* Site navigation uses native details; only the mobile drawer needs scripting. */
(() => {
  const toggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.wy-nav-side');
  if (!toggle || !sidebar) return;
  const mobile = window.matchMedia('(max-width: 768px)');
  const content = document.querySelector('.wy-nav-content-wrap');
  sidebar.setAttribute('aria-label', 'Site navigation');
  sidebar.querySelector('.wy-menu-vertical').removeAttribute('role');
  sidebar.querySelector('.wy-menu-vertical').removeAttribute('aria-label');
  function setOpen(open, restoreFocus = false) {
    document.body.classList.toggle('docs-nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    sidebar.inert = mobile.matches && !open;
    content.inert = mobile.matches && open;
    if (mobile.matches && open) {
      sidebar.setAttribute('role', 'dialog');
      sidebar.setAttribute('aria-modal', 'true');
    } else {
      sidebar.removeAttribute('role');
      sidebar.removeAttribute('aria-modal');
    }
    if (open) sidebar.querySelector('.nav-close').focus();
    else if (restoreFocus) toggle.focus();
  }
  toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
  sidebar.querySelector('.nav-close').addEventListener('click', () => setOpen(false, true));
  document.addEventListener('click', event => {
    if (mobile.matches && !sidebar.contains(event.target) && !toggle.contains(event.target)) setOpen(false);
  });
  document.addEventListener('keydown', event => {
    if (!mobile.matches || toggle.getAttribute('aria-expanded') !== 'true') return;
    if (event.key === 'Escape') { event.preventDefault(); setOpen(false, true); }
    if (event.key !== 'Tab') return;
    const focusable = [...sidebar.querySelectorAll('a[href], button, input, summary')]
      .filter(el => el.getClientRects().length && !el.disabled);
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  mobile.addEventListener('change', () => setOpen(false));
  setOpen(false);
})();
