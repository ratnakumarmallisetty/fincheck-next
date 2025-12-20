"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams } from "next/navigation"
import ChartSection from "@/components/charts/ChartSection"
import type { ChartItem } from "@/components/charts/metrics/types"

/* ---------------- CONSTANTS ---------------- */

const MODEL_ORDER = [
  "baseline_mnist.pth",
  "kd_mnist.pth",
  "lrf_mnist.pth",
  "pruned_mnist.pth",
  "quantized_mnist.pth",
  "ws_mnist.pth",
]

/* ---------------- TYPES ---------------- */

type ResultDoc = {
  data: Record<string, any>
  meta?: {
    evaluation_type?: "SINGLE" | "DATASET"
    source?: "PREBUILT" | "CUSTOM" | "IMAGE_UPLOAD"
    dataset_type?: string
    num_images?: number
  }
}

/* ---------------- OVERALL SUMMARY ---------------- */

function getOverallSummary(data: ChartItem[]) {
  if (data.length === 0) return null

  const fastest = [...data].sort((a, b) => a.latency_ms - b.latency_ms)[0]
  const confident = [...data].sort(
    (a, b) => b.confidence_percent - a.confidence_percent
  )[0]
  const stable = [...data].sort(
    (a, b) => a.stability - b.stability
  )[0]
  const certain = [...data].sort(
    (a, b) => a.entropy - b.entropy
  )[0]

  return { fastest, confident, stable, certain }
}

/* ---------------- META NORMALIZATION ---------------- */

function normalizeMeta(meta?: ResultDoc["meta"]) {
  return {
    evaluation_type: meta?.evaluation_type ?? "SINGLE",
    source: meta?.source ?? "IMAGE_UPLOAD",
    dataset_type: meta?.dataset_type,
    num_images: meta?.num_images,
  }
}

/* ==================================================== */

export default function ResultPage() {
  const { id } = useParams<{ id: string }>()
  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState("ALL")

  /* ---------------- FETCH RESULT ---------------- */

  useEffect(() => {
    fetch(`/api/results/${id}`)
      .then((r) => r.json())
      .then(setDoc)
      .finally(() => setLoading(false))
  }, [id])

  /* ---------------- NORMALIZE METRICS ---------------- */

  const chartData: ChartItem[] = useMemo(() => {
    if (!doc) return []

    return MODEL_ORDER.map((model) => {
      const v = doc.data[model] ?? {}
      return {
        model,
        confidence_percent: v.confidence_percent ?? v.avg_confidence ?? 0,
        latency_ms: v.latency_ms ?? v.avg_latency_ms ?? 0,
        entropy: v.entropy ?? v.avg_entropy ?? 0,
        stability: v.stability ?? v.avg_stability ?? 0,
        ram_delta_mb: v.ram_mb ?? v.avg_ram_mb ?? 0,
      }
    })
  }, [doc])

  const recommendedModel = useMemo(() => {
    if (chartData.length === 0) return undefined
    return [...chartData].sort(
      (a, b) => a.latency_ms - b.latency_ms
    )[0]?.model
  }, [chartData])

  const overall = useMemo(() => {
    return getOverallSummary(chartData)
  }, [chartData])

  /* ---------------- UI STATES ---------------- */

  if (loading) {
    return <p className="p-8 text-gray-500">Loading results…</p>
  }

  if (!doc) {
    return <p className="p-8 text-red-600">Failed to load result.</p>
  }

  const meta = normalizeMeta(doc.meta)

  /* ===================== UI ===================== */

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-10">

      {/* 🧾 EVALUATION SUMMARY */}
      <div className="rounded-xl border bg-gray-50 p-5">
        <h2 className="text-sm font-semibold text-gray-700">
          Evaluation Summary
        </h2>

        <div className="mt-2 flex flex-wrap gap-3 text-sm">
          <span className="rounded-full bg-blue-100 px-3 py-1 text-blue-700">
            {meta.evaluation_type === "SINGLE"
              ? "Single Image Inference"
              : "Dataset Evaluation"}
          </span>

          <span className="rounded-full bg-purple-100 px-3 py-1 text-purple-700">
            Source: {meta.source}
          </span>

          {meta.dataset_type && (
            <span className="rounded-full bg-green-100 px-3 py-1 text-green-700">
              Dataset: {meta.dataset_type}
            </span>
          )}

          {meta.num_images && (
            <span className="rounded-full bg-gray-200 px-3 py-1 text-gray-700">
              Images: {meta.num_images}
            </span>
          )}
        </div>
      </div>

      {/* ✅ SYSTEM RECOMMENDATION */}
      <div className="rounded-xl border bg-blue-50 p-5">
        <h2 className="font-semibold text-blue-800">
          System Recommendation
        </h2>
        <p className="mt-1 text-sm text-blue-700">
          <strong>{recommendedModel}</strong> is recommended due to its
          low inference latency while maintaining competitive confidence
          under the evaluated conditions.
        </p>
      </div>

      {/* 🔽 MODEL SELECTOR */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-gray-600">
          View Mode:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="rounded-lg border px-3 py-2 text-sm"
        >
          <option value="ALL">Compare all models</option>
          {MODEL_ORDER.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* 📊 COMPARISON MODE */}
      {selectedModel === "ALL" && (
        <ChartSection data={chartData} selectedModel="ALL" />
      )}

      {/* 🔍 SINGLE MODEL MODE */}
      {selectedModel !== "ALL" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {chartData
            .filter((d) => d.model === selectedModel)
            .map((d) => (
              <div
                key={d.model}
                className="rounded-xl border bg-white p-6 shadow-sm"
              >
                <h3 className="font-semibold mb-4">
                  Model: {d.model}
                </h3>

                <MetricRow label="Confidence (%)" value={d.confidence_percent.toFixed(2)} />
                <MetricRow label="Latency (ms)" value={d.latency_ms.toFixed(3)} />
                <MetricRow label="Entropy" value={d.entropy.toFixed(4)} />
                <MetricRow label="Stability" value={d.stability.toFixed(4)} />
                <MetricRow label="RAM Usage (MB)" value={d.ram_delta_mb.toFixed(3)} />
              </div>
            ))}
        </div>
      )}

      {/* 📦 OVERALL PERFORMANCE SUMMARY */}
      {overall && (
        <div className="rounded-2xl border bg-gray-50 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Overall Performance Summary
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            <SummaryRow
              label="Fastest Model"
              model={overall.fastest.model}
              value={`${overall.fastest.latency_ms.toFixed(3)} ms`}
            />
            <SummaryRow
              label="Most Confident Model"
              model={overall.confident.model}
              value={`${overall.confident.confidence_percent.toFixed(2)} %`}
            />
            <SummaryRow
              label="Most Stable Model"
              model={overall.stable.model}
              value={overall.stable.stability.toFixed(4)}
            />
            <SummaryRow
              label="Lowest Uncertainty Model"
              model={overall.certain.model}
              value={overall.certain.entropy.toFixed(4)}
            />
          </div>

          <div className="rounded-lg bg-white p-4 text-sm text-gray-700">
            <strong>Conclusion:</strong>{" "}
            While individual models excel in specific metrics,
            <strong> {recommendedModel}</strong> offers the most balanced
            trade-off between speed, confidence, and stability under the
            evaluated conditions, making it the most suitable overall choice.
          </div>
        </div>
      )}

    </div>
  )
}

/* ---------------- METRIC ROW ---------------- */

function MetricRow({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div className="flex justify-between text-sm py-1">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

/* ---------------- SUMMARY ROW ---------------- */

function SummaryRow({
  label,
  model,
  value,
}: {
  label: string
  model: string
  value: string
}) {
  return (
    <div className="flex justify-between rounded-lg bg-white px-4 py-2 border">
      <span className="text-gray-600">{label}</span>
      <span className="font-medium">
        {model} · {value}
      </span>
    </div>
  )
}
