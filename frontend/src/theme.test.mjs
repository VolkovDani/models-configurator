import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';

const storageKey = 'models-configurator-theme';

function startPage(storage) {
  const script = readFileSync(new URL('../../internal/status/theme.js', import.meta.url), 'utf8');
  const root = { dataset: {} };
  const picker = { hidden: true };
  const pageEvents = new EventTarget();
  const select = Object.assign(new EventTarget(), { value: 'system', parentElement: picker });
  runInNewContext(script, {
    document: {
      documentElement: root,
      addEventListener: pageEvents.addEventListener.bind(pageEvents),
      getElementById: (id) => (id === 'panel-theme' ? select : null),
    },
    localStorage: storage,
  });
  const appliedBeforeReady = root.dataset.theme;
  pageEvents.dispatchEvent(new Event('DOMContentLoaded'));
  return {
    root,
    select,
    picker,
    appliedBeforeReady,
    choose(value) {
      select.value = value;
      select.dispatchEvent(new Event('change'));
    },
  };
}

test('theme follows the system and persists an explicit choice across page loads', () => {
  const values = new Map([[storageKey, 'invalid preference']]);
  const storage = {
    getItem: (key) => values.get(key),
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
  const page = startPage(storage);
  assert.equal(page.picker.hidden, false);
  assert.equal(page.select.value, 'system');
  assert.equal(page.root.dataset.theme, undefined);

  page.choose('dark');
  assert.equal(page.root.dataset.theme, 'dark');
  const nextPage = startPage(storage);
  assert.equal(nextPage.appliedBeforeReady, 'dark');
  assert.equal(nextPage.select.value, 'dark');

  nextPage.choose('light');
  assert.equal(startPage(storage).appliedBeforeReady, 'light');
  nextPage.choose('system');
  assert.equal(nextPage.root.dataset.theme, undefined);
  assert.equal(values.has(storageKey), false);
});

test('theme can change for the current page when browser storage is blocked', () => {
  function denyAccess() {
    throw new Error('storage unavailable');
  }
  const page = startPage({ getItem: denyAccess, setItem: denyAccess });
  assert.equal(page.select.value, 'system');
  page.choose('dark');
  assert.equal(page.root.dataset.theme, 'dark');
});
