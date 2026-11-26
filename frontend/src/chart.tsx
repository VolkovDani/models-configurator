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
import { groupCalls, type Operation, type Point, readCalls, timeDomain } from './data.ts';

const colors: Record<string, string> = {
  chat: '#176957',
  embeddings: '#315ca5',
  decision: '#975915',
  bge_encode: '#8b4287',
};

type Labels = {
  title: string;
  profile: string;
  duration: string;
  unit: string;
  time: string;
};

function OperationPlot({
  series,
  dates,
  bounds,
  durationMax,
  timeTicks,
  numbers,
  labels,
}: {
  series: Operation;
  dates: { ticks: Intl.DateTimeFormat; tooltip: Intl.DateTimeFormat };
  bounds: [number, number];
  durationMax: number;
  timeTicks: number[];
  numbers: Intl.NumberFormat;
  labels: Labels;
}) {
  return (
    <figure className="history-plot">
      <figcaption>{series.label}</figcaption>
      <div className="history-plot-area">
        <ResponsiveContainer width="100%" height={164}>
          <ScatterChart margin={{ top: 8, right: 22, bottom: 5, left: 2 }} title={series.label}>
            <CartesianGrid stroke="#e1e7e1" vertical={false} />
            <XAxis
              dataKey="started"
              name={labels.time}
              type="number"
              scale="utc"
              domain={bounds}
              ticks={timeTicks}
              minTickGap={20}
              allowDataOverflow
              tickFormatter={(value: number) => dates.ticks.format(value)}
              tick={{ fontSize: 11, fill: '#445b52' }}
            />
            <YAxis
              dataKey="duration"
              name={labels.duration}
              type="number"
              domain={[0, durationMax]}
              width={55}
              tickCount={4}
              allowDecimals={false}
              tickFormatter={(value: number) => numbers.format(value)}
              tick={{ fontSize: 11, fill: '#445b52' }}
            />
            <Tooltip
              cursor={false}
              content={({ active, payload }) => {
                const point = payload[0]?.payload as Point | undefined;
                if (!active || !point) return null;
                return (
                  <div className="history-plot-tooltip" role="status">
                    <strong>
                      {numbers.format(point.duration)} {labels.unit}
                    </strong>
                    <span>{dates.tooltip.format(point.started)}</span>
                  </div>
                );
              }}
            />
            <Scatter
              name={series.label}
              data={series.points}
              fill={colors[series.operation] ?? '#445b52'}
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function HistoryChart({
  table,
  lang,
  labels,
}: {
  table: HTMLTableSectionElement;
  lang: string;
  labels: Labels;
}) {
  const [profiles, setProfiles] = useState(() => groupCalls(readCalls(table)));
  const [selectedId, select] = useState('');

  useEffect(() => {
    const update = () => setProfiles(groupCalls(readCalls(table)));
    const observer = new MutationObserver(update);
    observer.observe(table, { childList: true });
    update();
    return () => observer.disconnect();
  }, [table]);

  const profile = profiles.find((model) => model.id === selectedId) ?? profiles[0];
  if (!profile) return null;

  const [start, end] = timeDomain(profile.operations);
  const span = end - start;
  const padding = span * 0.04;
  const bounds: [number, number] = [start - padding, end + padding];
  const timeTicks = Array.from({ length: 4 }, (_, index) => start + (span * index) / 3);
  const dates = {
    ticks: new Intl.DateTimeFormat(lang, {
      timeZone: 'UTC',
      ...(span >= 86_400_000
        ? { month: 'short', day: 'numeric' }
        : { hour: '2-digit', minute: '2-digit', ...(span < 60_000 ? { second: '2-digit' } : {}) }),
    }),
    tooltip: new Intl.DateTimeFormat(lang, {
      timeZone: 'UTC',
      dateStyle: 'medium',
      timeStyle: 'medium',
    }),
  };
  const numbers = new Intl.NumberFormat(lang);
  let durationMax = 1;
  for (const { points } of profile.operations) {
    for (const { duration } of points) durationMax = Math.max(durationMax, duration);
  }

  return (
    <section className="history-chart" aria-labelledby="duration-chart-title">
      <div className="history-chart-header">
        <h2 id="duration-chart-title">{labels.title}</h2>
        <label>
          {labels.profile}
          <select value={profile.id} onChange={(event) => select(event.target.value)}>
            {profiles.map((model) => (
              <option key={model.id} value={model.id}>
                {profiles.some(({ id, name }) => name === model.name && id !== model.id)
                  ? `${model.name} · ${model.id.slice(-8)}`
                  : model.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="history-chart-axis">
        {labels.duration} ({labels.unit}) · {labels.time}
      </div>
      <div className="history-plots">
        {profile.operations.map((series) => (
          <OperationPlot
            key={series.operation}
            series={series}
            dates={dates}
            bounds={bounds}
            durationMax={durationMax}
            timeTicks={timeTicks}
            numbers={numbers}
            labels={labels}
          />
        ))}
      </div>
      <ul className="history-chart-legend">
        {profile.operations.map((series) => (
          <li key={series.operation}>
            <span
              aria-hidden="true"
              style={{ backgroundColor: colors[series.operation] ?? '#445b52' }}
            />
            {series.label} {series.endpoint && <code>{series.endpoint}</code>}
          </li>
        ))}
      </ul>
    </section>
  );
}

const mount = document.getElementById('duration-chart');
const table = document.getElementById('calls');
if (mount && table instanceof HTMLTableSectionElement) {
  const { title, chartProfile: profile, duration, unit, time } = mount.dataset;
  if (title && profile && duration && unit && time) {
    createRoot(mount).render(
      <HistoryChart
        table={table}
        lang={document.documentElement.lang}
        labels={{ title, profile, duration, unit, time }}
      />,
    );
  }
}
