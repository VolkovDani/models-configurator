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
  series: ProfileSeries[];
};

type ProfileSeries = { id: string; name: string; points: Point[] };

export function groupCalls(calls: ReadonlyArray<Call>): Operation[] {
  const operations = new Map<
    string,
    Omit<Operation, 'series'> & { profiles: Map<string, ProfileSeries> }
  >();
  for (const { modelId, profile, operation, label, endpoint, started, duration } of calls) {
    let group = operations.get(operation);
    if (!group) {
      group = { operation, label, endpoint, profiles: new Map() };
      operations.set(operation, group);
    }
    let series = group.profiles.get(modelId);
    if (!series) {
      series = { id: modelId, name: profile, points: [] };
      group.profiles.set(modelId, series);
    }
    series.points.push({ started, duration });
  }
  return Array.from(operations.values(), ({ profiles, ...operation }) => {
    const idsByName = new Map<string, string[]>();
    for (const { id, name } of profiles.values()) {
      const ids = idsByName.get(name);
      if (ids) ids.push(id);
      else idsByName.set(name, [id]);
    }
    return {
      ...operation,
      series: Array.from(profiles.values(), (series) => ({
        ...series,
        name: profileName(series, idsByName.get(series.name) ?? []),
        points: series.points.sort((a, b) => a.started - b.started),
      })),
    };
  });
}

function profileName({ id, name }: ProfileSeries, matchingIds: string[]) {
  if (matchingIds.length <= 1) return name;

  let length = 8;
  while (
    length < id.length &&
    matchingIds.some((otherId) => otherId !== id && otherId.endsWith(id.slice(-length)))
  ) {
    length += 1;
  }
  return `${name} · ${id.slice(-length)}`;
}

export function timeDomain(series: ReadonlyArray<Pick<ProfileSeries, 'points'>>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const { points } of series) {
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
