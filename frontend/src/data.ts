type CallUsage = {
  duration: number;
  inputTokens?: number;
  outputTokens?: number;
};
export type Point = CallUsage & { tokens: number };
export const tokenMetrics = ['inputTokens', 'outputTokens', 'totalTokens'] as const;
export type TokenMetric = (typeof tokenMetrics)[number];

export type Call = CallUsage & {
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

type ProfileSeries = { id: string; name: string; points: CallUsage[] };

export function groupCalls(calls: ReadonlyArray<Call>): Operation[] {
  const operations = new Map<
    string,
    Omit<Operation, 'series'> & { profiles: Map<string, ProfileSeries> }
  >();
  for (const {
    modelId,
    profile,
    operation,
    label,
    endpoint,
    duration,
    inputTokens,
    outputTokens,
  } of calls) {
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
    series.points.push({ duration, inputTokens, outputTokens });
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

export function seriesForMetric(series: ReadonlyArray<ProfileSeries>, metric: TokenMetric) {
  return series.flatMap(({ points, ...profile }, index) => {
    const measured = points.flatMap((point): Point[] => {
      const { inputTokens, outputTokens } = point;
      const tokens =
        metric === 'totalTokens'
          ? inputTokens !== undefined && outputTokens !== undefined
            ? inputTokens + outputTokens
            : undefined
          : point[metric];
      return tokens !== undefined && Number.isSafeInteger(tokens) ? [{ ...point, tokens }] : [];
    });
    return measured.length ? [{ ...profile, index, points: measured }] : [];
  });
}

function readCount(value: string | undefined): number | undefined {
  if (value === undefined || !/^\d+$/.test(value)) return undefined;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : undefined;
}

export function readCalls(table: HTMLTableSectionElement): Call[] {
  const calls: Call[] = [];
  for (const row of table.rows) {
    const {
      modelId,
      modelName: profile,
      operation,
      endpoint,
      durationMs,
      inputTokens,
      outputTokens,
    } = row.dataset;
    const duration = readCount(durationMs);
    if (!modelId || !profile || !operation || duration === undefined) continue;

    const label = row.querySelector('[data-operation-label]')?.textContent || operation;
    calls.push({
      modelId,
      profile,
      operation,
      label,
      endpoint: endpoint ?? '',
      duration,
      inputTokens: readCount(inputTokens),
      outputTokens: readCount(outputTokens),
    });
  }
  return calls;
}
