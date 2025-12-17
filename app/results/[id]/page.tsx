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
      confidence?: number
      latency_ms?: number
      throughput?: number
      entropy?: number
      stability?: number
      ram_mb?: number
      cold_start_ms?: number
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
        confidence: typeof v.confidence === "number" ? v.confidence : 0,
        latency_ms: typeof v.latency_ms === "number" ? v.latency_ms : 0,
        throughput: typeof v.throughput === "number" ? v.throughput : 0,
        entropy: typeof v.entropy === "number" ? v.entropy : 0,
        stability: typeof v.stability === "number" ? v.stability : 0,
        ram_mb: typeof v.ram_mb === "number" ? v.ram_mb : 0,
        cold_start_ms:
          typeof v.cold_start_ms === "number" ? v.cold_start_ms : 0,
      }
    })
  }, [doc])

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
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          Inference Results
        </h1>
        <p className="text-sm text-gray-500">
          Comparative analysis of optimized MNIST models across
          confidence, latency, throughput, entropy, stability,
          memory usage, and cold-start behavior
        </p>
      </header>

      {/* 📊 Charts */}
      {chartData.length > 0 && (
        <ChartSection data={chartData} />
      )}

      {/* 📄 Raw Output */}
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
