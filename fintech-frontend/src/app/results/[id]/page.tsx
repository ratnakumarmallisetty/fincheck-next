"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams } from "next/navigation"
import ChartSection from "../../../../components/charts/ChartSection"
import type { ChartItem } from "../../../../components/metrics/types"
import { ConfusionMatrix } from "../../../../components/confusion-matrix"
import type { ConfusionMatrixModel } from "../../../../components/confusion-matrix"


/* ================= CONSTANTS ================= */

const MODEL_ORDER = [
  "baseline_mnist.pth",
  "kd_mnist.pth",
  "lrf_mnist.pth",
  "pruned_mnist.pth",
  "quantized_mnist.pth",
  "ws_mnist.pth",
]

/* ================= TYPES ================= */

type ResultDoc = {
  data: Record<string, any>
  meta?: {
    evaluation_type?: "SINGLE" | "DATASET"
    source?: "PREBUILT" | "CUSTOM" | "IMAGE_UPLOAD"
    dataset_type?: string
    num_images?: number
  }
}

/* ================= HELPERS ================= */

function normalizeMeta(meta?: ResultDoc["meta"]) {
  return {
    evaluation_type: meta?.evaluation_type ?? "SINGLE",
    source: meta?.source ?? "IMAGE_UPLOAD",
    dataset_type: meta?.dataset_type,
    num_images: meta?.num_images,
  }
}

function scoreModel(m: ChartItem, datasetType?: string) {
  if (!datasetType) return -m.latency_ms

  if (datasetType.includes("NOISY") || datasetType.includes("BLUR")) {
    return -2 * m.entropy - 2 * m.stability + 0.5 * m.confidence_percent
  }

  return m.confidence_percent - m.entropy - 0.2 * m.latency_ms
}

function getOverallSummary(data: ChartItem[]) {
  if (!data.length) return null

  return {
    fastest: [...data].sort((a, b) => a.latency_ms - b.latency_ms)[0],
    confident: [...data].sort((a, b) => b.confidence_percent - a.confidence_percent)[0],
    stable: [...data].sort((a, b) => a.stability - b.stability)[0],
    certain: [...data].sort((a, b) => a.entropy - b.entropy)[0],
  }
}

/* ================= PAGE ================= */

export default function ResultPage() {
  const { id } = useParams<{ id: string }>()
  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState("ALL")

  /* -------- Fetch -------- */

  useEffect(() => {
    fetch(`/api/results/${id}`)
      .then((r) => r.json())
      .then(setDoc)
      .finally(() => setLoading(false))
  }, [id])

  const meta = normalizeMeta(doc?.meta)
  const isNoisy = meta.dataset_type?.includes("NOISY")

  /* -------- Normalize Data -------- */

  const chartData: ChartItem[] = useMemo(() => {
    if (!doc) return []

    return MODEL_ORDER.map((model) => {
      const v = doc.data[model] ?? {}

      return {
        model,

        confidence_percent: isNoisy ? v.confidence_mean ?? 0 : v.confidence_percent ?? 0,
        confidence_std: isNoisy ? v.confidence_std ?? 0 : 0,

        latency_ms: isNoisy ? v.latency_mean ?? 0 : v.latency_ms ?? 0,
        latency_std: isNoisy ? v.latency_std ?? 0 : 0,

        entropy: isNoisy ? v.entropy_mean ?? 0 : v.entropy ?? 0,
        entropy_std: isNoisy ? v.entropy_std ?? 0 : 0,

        stability: isNoisy ? v.stability_mean ?? 0 : v.stability ?? 0,
        stability_std: isNoisy ? v.stability_std ?? 0 : 0,

        ram_delta_mb: v.ram_mb ?? 0,
      }
    })
  }, [doc, isNoisy])

  const confusionMatrices = useMemo<ConfusionMatrixModel[]>(() => {
    if (!doc) return []

    return MODEL_ORDER.map((model) => {
      const evalData = doc.data?.[model]?.evaluation

      return {
        model,
        matrix: evalData?.confusion_matrix ?? [],
        FAR: evalData?.FAR,
        FRR: evalData?.FRR,
        risk_score: evalData?.risk_score
      }
    }).filter((m) => m.matrix.length > 0)
  }, [doc])

  /* -------- Best Model -------- */

  const recommendedModel = useMemo(() => {
    if (!chartData.length) return undefined
    return [...chartData].sort(
      (a, b) => scoreModel(b, meta.dataset_type) - scoreModel(a, meta.dataset_type)
    )[0].model
  }, [chartData, meta.dataset_type])

  const selectedModelData = useMemo(() => {
    if (selectedModel === "ALL") return null
    return chartData.find((m) => m.model === selectedModel) ?? null
  }, [selectedModel, chartData])

  const overall = useMemo(() => getOverallSummary(chartData), [chartData])

  /* -------- UI States -------- */

  if (loading) return <p className="p-8 text-gray-500">Loading results…</p>
  if (!doc) return <p className="p-8 text-red-600">Failed to load result.</p>

  /* ================= RENDER ================= */

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-10">

      {/* ===== Evaluation Summary ===== */}
      <div className="rounded-xl border bg-gray-50 p-5 space-y-3">
        <h2 className="text-sm font-semibold text-gray-700">
          Evaluation Summary
        </h2>

        {recommendedModel && (
          <div className="rounded-xl border bg-blue-50 p-5">
            <h2 className="font-semibold text-blue-800">
              🏆 System Recommendation
            </h2>
            <p className="mt-1 text-sm text-blue-700">
              <strong>{recommendedModel}</strong> is the best model for{" "}
              <strong>{meta.dataset_type ?? "this evaluation"}</strong>.
            </p>
          </div>
        )}
      </div>

      {/* ===== 🔽 MODEL SELECTOR (THIS IS THE DROPDOWN) ===== */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-gray-600">
          View Model:
        </label>
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
      </div>

      {/* ===== Charts ===== */}
      {selectedModel === "ALL" && (
        <ChartSection data={chartData} selectedModel="ALL" />
      )}

      {/* ===== Single Model Metrics ===== */}
      {selectedModelData && (
        <div
          className={`rounded-xl border p-6 ${
            selectedModelData.model === recommendedModel
              ? "bg-green-50 border-green-400"
              : "bg-white"
          }`}
        >
          <h3 className="font-semibold mb-4">
            Model: {selectedModelData.model}
          </h3>

          <MetricRow
            label="Confidence (%)"
            value={
              isNoisy
                ? `${selectedModelData.confidence_percent.toFixed(2)} ± ${(selectedModelData.confidence_std ?? 0).toFixed(2)}`
                : selectedModelData.confidence_percent.toFixed(2)
            }
          />

          <MetricRow
            label="Latency (ms)"
            value={
              isNoisy
                ? `${selectedModelData.latency_ms.toFixed(3)} ± ${(selectedModelData.latency_std ?? 0).toFixed(3)}`
                : selectedModelData.latency_ms.toFixed(3)
            }
          />

          <MetricRow
            label="Entropy"
            value={
              isNoisy
                ? `${selectedModelData.entropy.toFixed(4)} ± ${(selectedModelData.entropy_std ?? 0).toFixed(4)}`
                : selectedModelData.entropy.toFixed(4)
            }
          />

          <MetricRow
            label="Stability"
            value={
              isNoisy
                ? `${selectedModelData.stability.toFixed(4)} ± ${(selectedModelData.stability_std ?? 0).toFixed(4)}`
                : selectedModelData.stability.toFixed(4)
            }
          />
        </div>
      )}

      {/* ===== Confusion Matrices ===== */}
      { selectedModel === "ALL" && (
        <div className="space-y-8">
          {confusionMatrices.map((m) => (
            <ConfusionMatrix key={m.model} data={m} />
          ))}
        </div>
      )}

      { selectedModelData && (
        <ConfusionMatrix
          data={
            confusionMatrices.find(
              (m) => m.model === selectedModelData.model
            )!
          }
        />
      )}

      {/* ===== Overall Performance Summary ===== */}
      {overall && (
        <div className="rounded-2xl border bg-gray-50 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Overall Performance Summary
          </h2>

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
      )}
    </div>
  )
}

/* ================= UI HELPERS ================= */

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm py-1">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

function SummaryRow({ label, model, value }: { label: string; model: string; value: string }) {
  return (
    <div className="flex justify-between rounded-lg bg-white px-4 py-2 border">
      <span className="text-gray-600">{label}</span>
      <span className="font-medium">
        {model} · {value}
      </span>
    </div>
  )
}