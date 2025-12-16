"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams } from "next/navigation"
import ChartSection from "@/components/ChartSection"

type ModelResult = {
  prediction?: number
  confidence?: number
  latency_ms?: number
  ram_mb?: number
  cold_start?: boolean
}

type ResultDoc = {
  data: Record<string, ModelResult>
}

type ChartItem = {
  model: string
  prediction: number
  confidence: number
  latency_ms: number
  ram_mb: number
  throughput: number
  efficiency: number
  cold_start: boolean
}

export default function ResultPage() {
  const params = useParams()
  const id = params?.id as string

  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return

    fetch(`/api/results/${id}`)
      .then(async (r) => {
        if (!r.ok) throw new Error("Failed to fetch")
        return r.json()
      })
      .then((data) => {
        setDoc(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [id])

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

  /* ---------------- DATA TRANSFORMATION ---------------- */

  const chartData: ChartItem[] = useMemo(() => {
    const raw = Object.entries(doc.data).map(([model, v]) => {
      const confidence =
        typeof v.confidence === "number" ? v.confidence : 0
      const latency =
        typeof v.latency_ms === "number" ? v.latency_ms : 0
      const ram =
        typeof v.ram_mb === "number" ? v.ram_mb : 0

      return {
        model,
        prediction:
          typeof v.prediction === "number" ? v.prediction : -1,
        confidence,
        latency_ms: latency,
        ram_mb: ram,
        throughput:
          latency > 0 ? Number((1000 / latency).toFixed(2)) : 0,
        efficiency:
          latency > 0
            ? Number((confidence / latency).toFixed(2))
            : 0,
        cold_start: Boolean(v.cold_start),
      }
    })

    // Filter invalid models
    return raw.filter((m) => m.confidence > 0)
  }, [doc.data])

  /* ---------------- UI ---------------- */

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-12">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          Inference Results
        </h1>
        <p className="text-sm text-gray-500">
          Comparative analysis of optimized MNIST models across
          confidence, latency, memory, and efficiency metrics
        </p>
      </header>

      {/* 📊 Research-grade Charts */}
      {chartData.length > 0 && (
        <ChartSection data={chartData} />
      )}

      {/* 📄 Raw JSON output (debug / transparency) */}
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
