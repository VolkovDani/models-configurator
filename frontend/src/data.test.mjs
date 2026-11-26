import { deepStrictEqual } from 'node:assert';
import { test } from 'node:test';
import { groupCalls, readCalls, seriesForMetric } from './data.ts';

const call = (modelId, profile, operation, duration, inputTokens, outputTokens) => ({
  modelId,
  profile,
  operation,
  label: operation === 'embeddings' ? 'Векторизация' : 'Генерация',
  endpoint: `POST /v1/${operation}`,
  duration,
  inputTokens,
  outputTokens,
});

test('groups operations and profiles and plots available counts, including measured zero', () => {
  const [chat, embeddings] = groupCalls([
    call('one.rev', 'one', 'chat', 101, 20, 4),
    call('two.rev', 'two', 'embeddings', 0, 12),
    call('two.rev', 'two', 'chat', 40, undefined, 0),
    call('one.rev', 'one', 'chat', 28, 5),
    call('old.rev', 'old', 'chat', 500),
  ]);

  deepStrictEqual(
    [chat.operation, chat.label, chat.endpoint],
    ['chat', 'Генерация', 'POST /v1/chat'],
  );
  const chartPoints = (operation, metric) =>
    seriesForMetric(operation.series, metric).map(({ id, points }) => [
      id,
      points.map(({ tokens, duration }) => [tokens, duration]),
    ]);
  deepStrictEqual(chartPoints(chat, 'inputTokens'), [
    [
      'one.rev',
      [
        [20, 101],
        [5, 28],
      ],
    ],
  ]);
  deepStrictEqual(chartPoints(chat, 'outputTokens'), [
    ['one.rev', [[4, 101]]],
    ['two.rev', [[0, 40]]],
  ]);
  deepStrictEqual(chartPoints(chat, 'totalTokens'), [['one.rev', [[24, 101]]]]);
  deepStrictEqual(chartPoints(embeddings, 'inputTokens'), [['two.rev', [[12, 0]]]]);
  deepStrictEqual(chartPoints(embeddings, 'totalTokens'), []);
});

test('disambiguates equal profile names with distinct revision suffixes', () => {
  const [operation] = groupCalls([
    call('example.bge.111122223333', 'bge', 'embeddings', 30, 3),
    call('example.bge.111122224444', 'bge', 'embeddings', 20, 2),
    call('example.bge.aa9922223333', 'bge', 'embeddings', 10, 1),
  ]);

  deepStrictEqual(
    operation.series.map(({ name }) => name),
    ['bge · 122223333', 'bge · 22224444', 'bge · 922223333'],
  );
});

test('appending older rows adds points without reordering operations or profiles', () => {
  const calls = [
    call('slow.rev', 'slow', 'embeddings', 30, 3),
    call('fast.rev', 'fast', 'embeddings', 40, 4),
  ];
  const updated = groupCalls([
    ...calls,
    call('extra.rev', 'extra', 'chat', 20, 2, 1),
    call('fast.rev', 'fast', 'embeddings', 10, 1),
  ]);

  deepStrictEqual(
    updated.map(({ operation }) => operation),
    ['embeddings', 'chat'],
  );
  deepStrictEqual(
    updated[0].series.map(({ id }) => id),
    ['slow.rev', 'fast.rev'],
  );
  deepStrictEqual(
    updated[0].series[1].points.map(({ inputTokens }) => inputTokens),
    [4, 1],
  );
});

test('reads counts from history rows without reading timestamps', () => {
  const row = (fields) => ({
    dataset: {
      modelId: 'profile',
      modelName: 'Model',
      operation: 'chat',
      durationMs: '0',
      ...fields,
    },
    querySelector: (selector) => {
      deepStrictEqual(selector, '[data-operation-label]');
      return { textContent: 'Генерация' };
    },
  });
  const calls = readCalls({
    rows: [
      row({ inputTokens: '28', outputTokens: '0' }),
      row({}),
      row({ inputTokens: '-4', outputTokens: '2.5' }),
      row({ inputTokens: '1', outputTokens: '' }),
    ],
  });
  deepStrictEqual(
    calls.map(({ duration, inputTokens, outputTokens }) => [duration, inputTokens, outputTokens]),
    [
      [0, 28, 0],
      [0, undefined, undefined],
      [0, undefined, undefined],
      [0, 1, undefined],
    ],
  );
});
