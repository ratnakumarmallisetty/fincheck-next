"use client"

import { ChartItem } from "../charts/metrics/types"
import MetricBar from "./metrics/MetricBar"

export default function ChartSection({
  data,
}: {
  data: ChartItem[]
}) {
  return (
    <div className="space-y-10">
      <MetricBar title="Confidence (%)" dataKey="confidence" data={data} />
      <MetricBar title="Latency (ms)" dataKey="latency_ms" data={data} />
      <MetricBar title="Throughput (samples/sec)" dataKey="throughput" data={data} />
      <MetricBar title="Entropy" dataKey="entropy" data={data} />
      <MetricBar title="Stability" dataKey="stability" data={data} />
      <MetricBar title="RAM Usage (MB)" dataKey="ram_mb" data={data} />
      <MetricBar title="Cold Start Time (ms)" dataKey="cold_start_ms" data={data} />
    </div>
  )
}
