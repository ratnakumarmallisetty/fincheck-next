"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import GraphCard from "../GraphCard"
import { ChartItem } from "./types"

export default function MetricBar({
  title,
  dataKey,
  data,
}: {
  title: string
  dataKey: keyof ChartItem
  data: ChartItem[]
}) {
  return (
    <GraphCard title={title}>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis />
          <Tooltip />
          <Bar dataKey={dataKey as string} />
        </BarChart>
      </ResponsiveContainer>
    </GraphCard>
  )
}
