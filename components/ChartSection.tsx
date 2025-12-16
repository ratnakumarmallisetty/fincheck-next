"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  CartesianGrid,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts"
import GraphCard from "./GraphCard"

type ChartItem = {
  model: string
  prediction: number
  confidence: number
  latency_ms: number
  ram_mb: number
  throughput: number
  cold_start?: boolean
}

export default function ChartSection({ data }: { data: ChartItem[] }) {
  const maxLatency = Math.max(...data.map(d => d.latency_ms))
  const maxRam = Math.max(...data.map(d => d.ram_mb))

  /* Composite performance score (normalized) */
  const performance = data.map(d => ({
    model: d.model,
    score: +(
      d.confidence * 0.5 +
      (1 - d.latency_ms / maxLatency) * 30 +
      (1 - d.ram_mb / maxRam) * 20
    ).toFixed(2),
  }))

  /* Pareto-optimal models */
  const pareto = data.filter(a =>
    !data.some(b =>
      b.confidence >= a.confidence &&
      b.latency_ms <= a.latency_ms &&
      (b.confidence > a.confidence || b.latency_ms < a.latency_ms)
    )
  )

  return (
    <div className="space-y-12">

      {/* BASIC METRICS */}
      <div className="grid gap-6 md:grid-cols-2">
        <MetricBar title="Confidence (%)" data={data} keyName="confidence" />
        <MetricBar title="Latency (ms)" data={data} keyName="latency_ms" />
        <MetricBar title="Memory Usage (MB)" data={data} keyName="ram_mb" />
        <MetricBar title="Throughput (inferences/sec)" data={data} keyName="throughput" />
      </div>

      {/* TRADE-OFF ANALYSIS */}
      <div className="grid gap-6 md:grid-cols-2">
        <ScatterGraph
          title="Confidence vs Latency Trade-off"
          x="latency_ms"
          y="confidence"
          data={data}
        />

        <ScatterGraph
          title="Memory vs Latency Efficiency"
          x="ram_mb"
          y="latency_ms"
          data={data}
        />
      </div>

      {/* COMPOSITE PERFORMANCE */}
      <GraphCard title="Composite Performance Index (Normalized)">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={performance}>
            <XAxis dataKey="model" hide />
            <YAxis />
            <Tooltip />
            <Bar dataKey="score" />
          </BarChart>
        </ResponsiveContainer>
      </GraphCard>

      {/* RADAR COMPARISON */}
      <GraphCard title="Multi-Metric Radar Comparison">
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="model" />
            <PolarRadiusAxis />
            <Radar
              name="Confidence"
              dataKey="confidence"
              stroke="#000"
              fill="#000"
              fillOpacity={0.2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </GraphCard>

      {/* PARETO FRONT */}
      <GraphCard title="Pareto-Optimal Models (Efficiency Frontier)">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid />
            <XAxis dataKey="latency_ms" />
            <YAxis dataKey="confidence" />
            <Tooltip />
            <Scatter data={pareto} />
          </ScatterChart>
        </ResponsiveContainer>
      </GraphCard>

      {/* COLD VS WARM */}
      <GraphCard title="Cold vs Warm Start Impact">
        <div className="flex flex-wrap gap-3">
          {data.map(m => (
            <span
              key={m.model}
              className={`rounded-full px-4 py-2 text-sm font-medium ${
                m.cold_start
                  ? "bg-red-100 text-red-700"
                  : "bg-green-100 text-green-700"
              }`}
            >
              {m.model} — {m.cold_start ? "Cold Start" : "Warm Start"}
            </span>
          ))}
        </div>
      </GraphCard>
    </div>
  )
}

/* ---------- Reusable sub-components ---------- */

function MetricBar({
  title,
  data,
  keyName,
}: {
  title: string
  data: any[]
  keyName: string
}) {
  return (
    <GraphCard title={title}>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis />
          <Tooltip />
          <Bar dataKey={keyName} />
        </BarChart>
      </ResponsiveContainer>
    </GraphCard>
  )
}

function ScatterGraph({
  title,
  x,
  y,
  data,
}: {
  title: string
  x: string
  y: string
  data: any[]
}) {
  return (
    <GraphCard title={title}>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid />
          <XAxis dataKey={x} />
          <YAxis dataKey={y} />
          <Tooltip />
          <Scatter data={data} />
        </ScatterChart>
      </ResponsiveContainer>
    </GraphCard>
  )
}
