"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams } from "next/navigation"
import ChartSection from "@/components/charts/ChartSection"
import type { ChartItem } from "@/components/charts/metrics/types"

/* ---------------- BACKEND DOCUMENT TYPE ---------------- */

type ResultDoc = {
  data: Record<
    string,
    {
      confidence_percent?: number
      latency_ms?: number
      entropy?: number
      stability?: number
      ram_mb?: number
    }
  >
}

/* ---------------- FIXED MODEL ORDER ---------------- */

const MODEL_ORDER = [
  "baseline_mnist.pth",
  "kd_mnist.pth",
  "lrf_mnist.pth",
  "pruned_mnist.pth",
  "quantized_mnist.pth",
  "ws_mnist.pth",
]

export default function ResultPage() {
  const params = useParams()
  const id = typeof params?.id === "string" ? params.id : null

  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState("ALL")

  /* ---------------- FETCH ---------------- */

  useEffect(() => {
    if (!id) return

    fetch(`/api/results/${id}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to fetch")
        return r.json()
      })
      .then((data) => {
        setDoc(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [id])

  /* ---------------- DATA → CHART MODEL ---------------- */

  const chartData: ChartItem[] = useMemo(() => {
    if (!doc?.data) return []

    return MODEL_ORDER.map((model) => {
      const v = doc.data[model] ?? {}

      return {
        model,
        confidence_percent: v.confidence_percent ?? 0,
        latency_ms: v.latency_ms ?? 0,
        entropy: v.entropy ?? 0,
        stability: v.stability ?? 0,
        ram_delta_mb: v.ram_mb ?? 0,
      }
    })
  }, [doc])

  const selectedData = chartData.find(
    (d) => d.model === selectedModel
  )

  /* ---------------- UI STATES ---------------- */

  if (loading) {
    return (
      <p className="p-8 text-gray-500">
        Loading inference results…
      </p>
    )
  }

  if (!doc || !doc.data) {
    return (
      <p className="p-8 text-red-600">
        Failed to load results
      </p>
    )
  }

  /* ---------------- UI ---------------- */

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-12">
      {/* ---------- HEADER ---------- */}
      <header className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">
          Inference Results
        </h1>
        <p className="text-sm text-gray-500">
          Comparative analysis of optimized MNIST models
          using single-image web inference metrics.
        </p>

        {/* MODEL SELECTOR */}
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="rounded-lg border px-3 py-2 text-sm"
        >
          <option value="ALL">Compare all models</option>
          {MODEL_ORDER.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </header>

      {/* ---------- COMPARISON MODE ---------- */}
      {selectedModel === "ALL" && (
        <ChartSection
          data={chartData}
          selectedModel={selectedModel}
        />
      )}

      {/* ---------- SINGLE MODEL MODE ---------- */}
      {selectedModel !== "ALL" && selectedData && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 animate-fade-in">
          <MetricCard label="Latency (ms)" value={selectedData.latency_ms} />
          <MetricCard label="Confidence (%)" value={selectedData.confidence_percent} />
          <MetricCard label="Entropy" value={selectedData.entropy} />
          <MetricCard label="Stability" value={selectedData.stability} />
          <MetricCard label="RAM Usage (MB)" value={selectedData.ram_delta_mb} />
        </div>
      )}

      {/* ---------- RAW OUTPUT (RESTORED) ---------- */}
      <div className="rounded-2xl border bg-gray-50 p-6">
        <h2 className="mb-3 text-lg font-semibold">
          Raw Model Output
        </h2>
        <pre className="overflow-auto rounded-lg bg-white p-4 text-sm text-gray-800">
          {JSON.stringify(doc.data, null, 2)}
        </pre>
      </div>
    </div>
  )
}

/* ---------------- METRIC CARD ---------------- */

function MetricCard({
  label,
  value,
}: {
  label: string
  value: number
}) {
  return (
    <div className="rounded-xl border bg-white p-5 shadow-sm">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold">
        {value.toFixed(3)}
      </p>
    </div>
  )
}
