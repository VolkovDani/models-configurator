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
import { groupCalls, type Point, readCalls, timeDomain } from './data.ts';

const colors = ['#176957', '#315ca5', '#975915', '#8b4287', '#a83b35', '#067282'];

type Labels = {
  title: string;
  usage: string;
  duration: string;
  unit: string;
  time: string;
};

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

  const profiles = operation.series.map((profile, index) => ({
    ...profile,
    scatterId: `history-profile-${profile.id}`,
    color: colors[index] ?? `hsl(${(index * 137.508) % 360} 60% 36%)`,
  }));
  const [start, end] = timeDomain(profiles);
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
  for (const { points } of profiles) {
    for (const { duration } of points) durationMax = Math.max(durationMax, duration);
  }

  return (
    <section className="history-chart" aria-labelledby="duration-chart-title">
      <div className="history-chart-header">
        <h2 id="duration-chart-title">{labels.title}</h2>
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
      </div>
      <div className="history-chart-axis">
        {labels.duration} ({labels.unit}) · {labels.time}
      </div>
      <div className="history-plot-area">
        <ResponsiveContainer width="100%" height={164}>
          <ScatterChart margin={{ top: 8, right: 22, bottom: 5, left: 2 }} title={operation.label}>
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
                    <span>{dates.tooltip.format(point.started)}</span>
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
    </section>
  );
}

const mount = document.getElementById('duration-chart');
const table = document.getElementById('calls');
if (mount && table instanceof HTMLTableSectionElement) {
  const { title, chartUsage: usage, duration, unit, time } = mount.dataset;
  if (title && usage && duration && unit && time) {
    createRoot(mount).render(
      <HistoryChart
        table={table}
        lang={document.documentElement.lang}
        labels={{ title, usage, duration, unit, time }}
      />,
    );
  }
}
