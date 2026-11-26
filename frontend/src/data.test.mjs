import { deepStrictEqual, strictEqual } from 'node:assert';
import { test } from 'node:test';
import { groupCalls, timeDomain } from './data.ts';

test('separates pinned model profiles and operation series, ordering individual calls by time', () => {
  const call = (modelId, operation, started, duration) => ({
    modelId,
    profile: 'bge',
    operation,
    label: operation,
    endpoint: `POST /v1/${operation}`,
    started,
    duration,
  });
  const profiles = groupCalls([
    call('rev1', 'embeddings', 3000, 30),
    call('rev1', 'bge_encode', 2000, 0),
    call('rev2', 'bge_encode', 2000, 40),
    call('rev1', 'embeddings', 1000, 10),
  ]);

  deepStrictEqual(
    profiles.map((model) => [model.id, model.operations.map((series) => series.operation)]),
    [
      ['rev1', ['embeddings', 'bge_encode']],
      ['rev2', ['bge_encode']],
    ],
  );
  const [embeddings, encode] = profiles[0].operations;
  deepStrictEqual(
    embeddings.points.map((point) => [point.started, point.duration]),
    [
      [1000, 10],
      [3000, 30],
    ],
  );
  strictEqual(encode.points[0].duration, 0);
  strictEqual(encode.endpoint, 'POST /v1/bge_encode');
  deepStrictEqual(timeDomain(profiles[0].operations), [1000, 3000]);
});

test('gives a single timestamp a visible time range', () => {
  const [from, to] = timeDomain([{ points: [{ started: 10000, duration: 0 }] }]);
  strictEqual(from < 10000 && to > 10000, true);
});
