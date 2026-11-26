export type Point = { started: number; duration: number };

export type Call = Point & {
  modelId: string;
  profile: string;
  operation: string;
  label: string;
  endpoint: string;
};

export type Operation = {
  operation: string;
  label: string;
  endpoint: string;
  points: Point[];
};

export type Profile = { id: string; name: string; operations: Operation[] };

export function groupCalls(calls: ReadonlyArray<Call>): Profile[] {
  const models = new Map<string, { name: string; operations: Map<string, Operation> }>();
  for (const { modelId, profile, operation, label, endpoint, started, duration } of calls) {
    let model = models.get(modelId);
    if (!model) {
      model = { name: profile, operations: new Map() };
      models.set(modelId, model);
    }
    let series = model.operations.get(operation);
    if (!series) {
      series = { operation, label, endpoint, points: [] };
      model.operations.set(operation, series);
    }
    series.points.push({ started, duration });
  }
  return Array.from(models, ([id, model]) => ({
    id,
    name: model.name,
    operations: Array.from(model.operations.values(), (series) => ({
      ...series,
      points: series.points.sort((a, b) => a.started - b.started),
    })),
  }));
}

export function timeDomain(operations: ReadonlyArray<Pick<Operation, 'points'>>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const { points } of operations) {
    for (const { started } of points) {
      min = Math.min(min, started);
      max = Math.max(max, started);
    }
  }
  return min === max ? [min - 1000, max + 1000] : [min, max];
}

export function readCalls(table: HTMLTableSectionElement): Call[] {
  const calls: Call[] = [];
  for (const row of table.rows) {
    const { modelId, modelName: profile, operation, endpoint, durationMs } = row.dataset;
    const started = Date.parse(row.querySelector('time')?.dateTime ?? '');
    const duration = durationMs === undefined ? NaN : Number(durationMs);
    if (
      !modelId ||
      !profile ||
      !operation ||
      !Number.isFinite(started) ||
      !Number.isFinite(duration)
    ) {
      continue;
    }
    const label = row.querySelector('[data-operation-label]')?.textContent || operation;
    calls.push({ modelId, profile, operation, label, endpoint: endpoint ?? '', started, duration });
  }
  return calls;
}
