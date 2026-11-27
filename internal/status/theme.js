(() => {
  const storageKey = 'models-configurator-theme';
  const root = document.documentElement;

  function apply(value) {
    if (value === 'light' || value === 'dark') {
      root.dataset.theme = value;
      return value;
    }
    delete root.dataset.theme;
    return 'system';
  }

  let saved;
  try {
    saved = localStorage.getItem(storageKey);
  } catch {
    // Use the system preference when browser storage is unavailable.
  }
  const initial = apply(saved);

  document.addEventListener('DOMContentLoaded', () => {
    const select = document.getElementById('panel-theme');
    if (!select) return;
    select.value = initial;
    select.parentElement.hidden = false;
    select.addEventListener('change', () => {
      const choice = apply(select.value);
      try {
        if (choice === 'system') localStorage.removeItem(storageKey);
        else localStorage.setItem(storageKey, choice);
      } catch {
        // Keep the selection active for this page even if it cannot be saved.
      }
    });
  });
})();
