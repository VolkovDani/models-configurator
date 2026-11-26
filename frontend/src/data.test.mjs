import { deepStrictEqual, strictEqual } from 'node:assert';
import { test } from 'node:test';
import { groupCalls, timeDomain } from './data.ts';

const call = (modelId, profile, operation, started, duration) => ({
  modelId,
  profile,
  operation,
  label: operation === 'embeddings' ? 'Векторизация' : 'Кодирование BGE',
  endpoint: `POST /v1/${operation}`,
  started,
  duration,
});

test('groups recorded operations into shared charts with separate pinned profiles', () => {
  const operations = groupCalls([
    call('slow.rev', 'slow', 'embeddings', 3000, 30),
    call('slow.rev', 'slow', 'bge_encode', 4000, 0),
    call('fast.rev', 'fast', 'embeddings', 2000, 40),
    call('slow.rev', 'slow', 'embeddings', 1000, 10),
  ]);

  deepStrictEqual(
    operations.map(({ operation, label, endpoint }) => [operation, label, endpoint]),
    [
      ['embeddings', 'Векторизация', 'POST /v1/embeddings'],
      ['bge_encode', 'Кодирование BGE', 'POST /v1/bge_encode'],
    ],
  );
  const [embeddings, encode] = operations;
  deepStrictEqual(embeddings.series, [
    {
      id: 'slow.rev',
      name: 'slow',
      points: [
        { started: 1000, duration: 10 },
        { started: 3000, duration: 30 },
      ],
    },
    { id: 'fast.rev', name: 'fast', points: [{ started: 2000, duration: 40 }] },
  ]);
  deepStrictEqual(encode.series, [
    { id: 'slow.rev', name: 'slow', points: [{ started: 4000, duration: 0 }] },
  ]);
  deepStrictEqual(timeDomain(embeddings.series), [1000, 3000]);
});

test('disambiguates equal profile names with distinct revision suffixes', () => {
  const [operation] = groupCalls([
    call('example.bge.111122223333', 'bge', 'embeddings', 3000, 30),
    call('example.bge.111122224444', 'bge', 'embeddings', 2000, 20),
    call('example.bge.aa9922223333', 'bge', 'embeddings', 1000, 10),
  ]);

  deepStrictEqual(
    operation.series.map(({ name }) => name),
    ['bge · 122223333', 'bge · 22224444', 'bge · 922223333'],
  );
});

test('appending older calls and new operations retains existing operation keys and profile order', () => {
  const calls = [
    call('slow.rev', 'slow', 'embeddings', 3000, 30),
    call('fast.rev', 'fast', 'embeddings', 2000, 40),
  ];
  const updated = groupCalls([
    ...calls,
    call('extra.rev', 'extra', 'bge_encode', 1000, 20),
    call('fast.rev', 'fast', 'embeddings', 1000, 10),
  ]);

  deepStrictEqual(
    updated.map(({ operation }) => operation),
    ['embeddings', 'bge_encode'],
  );
  deepStrictEqual(
    updated[0].series.map(({ id }) => id),
    ['slow.rev', 'fast.rev'],
  );
  strictEqual(updated[0].series[1].points[0].started, 1000);
});

test('gives a single timestamp a visible time range', () => {
  const [from, to] = timeDomain([{ points: [{ started: 10000, duration: 0 }] }]);
  strictEqual(from < 10000 && to > 10000, true);
});
