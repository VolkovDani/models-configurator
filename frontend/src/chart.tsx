import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  groupCalls,
  type Point,
  readCalls,
  seriesForMetric,
  type TokenMetric,
  tokenMetrics,
} from './data.ts';

const colors = ['#176957', '#315ca5', '#975915', '#8b4287', '#a83b35', '#067282'];

type Labels = {
  title: string;
  usage: string;
  duration: string;
  unit: string;
  tokenMetric: string;
  noTokenData: string;
} & Record<TokenMetric, string>;

function HistoryChart({
  table,
  lang,
  labels,
}: {
  table: HTMLTableSectionElement;
  lang: string;
  labels: Labels;
}) {
  const [operations, setOperations] = useState(() => groupCalls(readCalls(table)));
  const [selectedOperation, selectOperation] = useState('');
  const [selectedMetric, selectMetric] = useState<TokenMetric>('inputTokens');

  useEffect(() => {
    const update = () => setOperations(groupCalls(readCalls(table)));
    const observer = new MutationObserver(update);
    observer.observe(table, { childList: true });
    update();
    return () => observer.disconnect();
  }, [table]);

  const operation =
    operations.find((item) => item.operation === selectedOperation) ?? operations[0];
  if (!operation) return null;

  const available = tokenMetrics
    .map((metric) => ({ metric, series: seriesForMetric(operation.series, metric) }))
    .filter(({ series }) => series.length > 0);
  const selected = available.find(({ metric }) => metric === selectedMetric) ?? available[0];
  const profiles = (selected?.series ?? []).map((profile) => ({
    ...profile,
    scatterId: `history-profile-${profile.id}`,
    color: colors[profile.index] ?? `hsl(${(profile.index * 137.508) % 360} 60% 36%)`,
  }));
  const numbers = new Intl.NumberFormat(lang);
  let durationMax = 0;
  let tokensMax = 0;
  for (const { points } of profiles) {
    for (const { duration, tokens } of points) {
      durationMax = Math.max(durationMax, duration);
      tokensMax = Math.max(tokensMax, tokens);
    }
  }

  return (
    <section className="history-chart" aria-labelledby="duration-chart-title">
      <div className="history-chart-header">
        <h2 id="duration-chart-title">{labels.title}</h2>
        <div className="history-chart-filters">
          <label>
            {labels.usage}
            <select
              value={operation.operation}
              onChange={(event) => selectOperation(event.target.value)}
            >
              {operations.map((item) => (
                <option key={item.operation} value={item.operation}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          {selected && available.length > 1 && (
            <label>
              {labels.tokenMetric}
              <select
                value={selected.metric}
                onChange={(event) => {
                  const metric = tokenMetrics.find((item) => item === event.target.value);
                  if (metric) selectMetric(metric);
                }}
              >
                {available.map(({ metric }) => (
                  <option key={metric} value={metric}>
                    {labels[metric]}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </div>
      {selected ? (
        <>
          <div className="history-chart-axis">
            {labels.duration} ({labels.unit})
          </div>
          <div className="history-plot-area">
            <ResponsiveContainer width="100%" height={188}>
              <ScatterChart
                margin={{ top: 8, right: 22, bottom: 5, left: 2 }}
                title={operation.label}
              >
                <CartesianGrid stroke="#e1e7e1" vertical={false} />
                <XAxis
                  dataKey="tokens"
                  name={labels[selected.metric]}
                  type="number"
                  domain={[0, tokensMax + Math.max(1, Math.ceil(tokensMax * 0.04))]}
                  tickCount={5}
                  allowDecimals={false}
                  minTickGap={20}
                  tickFormatter={(value: number) => numbers.format(value)}
                  tick={{ fontSize: 11, fill: '#445b52' }}
                />
                <YAxis
                  dataKey="duration"
                  name={labels.duration}
                  type="number"
                  domain={[0, durationMax + Math.max(1, Math.ceil(durationMax * 0.04))]}
                  width={55}
                  tickCount={4}
                  allowDecimals={false}
                  tickFormatter={(value: number) => numbers.format(value)}
                  tick={{ fontSize: 11, fill: '#445b52' }}
                />
                <Tooltip
                  cursor={false}
                  content={({ active, payload }) => {
                    const entry = payload[0];
                    const point = entry?.payload as Point | undefined;
                    if (!active || !point) return null;
                    const profile = profiles.find(
                      ({ scatterId }) => scatterId === entry.graphicalItemId,
                    );
                    return (
                      <div className="history-plot-tooltip" role="status">
                        {profile && <span>{profile.name}</span>}
                        <strong>
                          {numbers.format(point.duration)} {labels.unit}
                        </strong>
                        {point.inputTokens !== undefined && (
                          <span>
                            {labels.inputTokens} · {numbers.format(point.inputTokens)}
                          </span>
                        )}
                        {point.outputTokens !== undefined && (
                          <span>
                            {labels.outputTokens} · {numbers.format(point.outputTokens)}
                          </span>
                        )}
                        {selected.metric === 'totalTokens' && (
                          <span>
                            {labels.totalTokens} · {numbers.format(point.tokens)}
                          </span>
                        )}
                      </div>
                    );
                  }}
                />
                {profiles.map(({ id, scatterId, name, points, color }) => (
                  <Scatter
                    key={id}
                    id={scatterId}
                    name={name}
                    data={points}
                    fill={color}
                    isAnimationActive={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="history-chart-axis history-chart-x-axis">{labels[selected.metric]}</div>
          <ul className="history-chart-legend">
            {operation.endpoint && (
              <li>
                <code>{operation.endpoint}</code>
              </li>
            )}
            {profiles.map(({ id, name, color }) => (
              <li key={id}>
                <span aria-hidden="true" style={{ backgroundColor: color }} />
                {name}
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p>{labels.noTokenData}</p>
      )}
    </section>
  );
}

const mount = document.getElementById('duration-chart');
const table = document.getElementById('calls');
if (mount && table instanceof HTMLTableSectionElement) {
  const {
    title,
    chartUsage: usage,
    duration,
    unit,
    tokenMetric,
    inputTokens,
    outputTokens,
    totalTokens,
    noTokenData,
  } = mount.dataset;
  if (
    title &&
    usage &&
    duration &&
    unit &&
    tokenMetric &&
    inputTokens &&
    outputTokens &&
    totalTokens &&
    noTokenData
  ) {
    createRoot(mount).render(
      <HistoryChart
        table={table}
        lang={document.documentElement.lang}
        labels={{
          title,
          usage,
          duration,
          unit,
          tokenMetric,
          inputTokens,
          outputTokens,
          totalTokens,
          noTokenData,
        }}
      />,
    );
  }
}
