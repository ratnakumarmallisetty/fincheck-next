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

const SAFE_THRESHOLD = 0.15

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

/* ================= PAGE ================= */

export default function ResultPage() {
  const { id } = useParams<{ id: string }>()

  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)

  const [selectedModel, setSelectedModel] = useState("ALL")
  const [sortBy, setSortBy] = useState<
    "latency" | "confidence" | "risk" | "balanced"
  >("balanced")

  useEffect(() => {
    fetch(`/api/results/${id}`)
      .then((r) => r.json())
      .then(setDoc)
      .finally(() => setLoading(false))
  }, [id])

  const meta = normalizeMeta(doc?.meta)
  const isNoisy = meta.dataset_type?.includes("NOISY")

  /* ================= NORMALIZE CHART DATA ================= */

  const chartData = useMemo(() => {
    if (!doc) return []

    return MODEL_ORDER.map((model) => {
      const v = doc.data?.[model] ?? {}

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
        risk_score: doc?.data?.[model]?.evaluation?.risk_score ?? 999,
      }
    })
  }, [doc, isNoisy])

  /* ================= CONFUSION MATRICES ================= */

  const confusionMatrices = useMemo<ConfusionMatrixModel[]>(() => {
    if (!doc) return []

    return MODEL_ORDER.map((model) => {
      const evalData = doc.data?.[model]?.evaluation
      return {
        model,
        matrix: evalData?.confusion_matrix ?? [],
        FAR: evalData?.FAR,
        FRR: evalData?.FRR,
        risk_score: evalData?.risk_score,
      }
    }).filter((m) => m.matrix.length > 0)
  }, [doc])

  /* ================= SORTING ================= */

  const sortedChartData = useMemo(() => {
    const arr = [...chartData]

    if (sortBy === "latency")
      return arr.sort((a, b) => a.latency_ms - b.latency_ms)

    if (sortBy === "confidence")
      return arr.sort((a, b) => b.confidence_percent - a.confidence_percent)

    if (sortBy === "risk")
      return arr.sort((a, b) => a.risk_score - b.risk_score)

    return arr.sort(
      (a, b) =>
        scoreModel(b, meta.dataset_type) -
        scoreModel(a, meta.dataset_type)
    )
  }, [chartData, sortBy, meta.dataset_type])

  /* ================= SPECIAL PICKS ================= */

  const safestModel = [...chartData].sort((a, b) => a.risk_score - b.risk_score)[0]

  const fastestSafeModel = [...chartData]
    .filter((m) => m.risk_score < SAFE_THRESHOLD)
    .sort((a, b) => a.latency_ms - b.latency_ms)[0]

  const balancedModel = [...chartData].sort(
    (a, b) =>
      scoreModel(b, meta.dataset_type) -
      scoreModel(a, meta.dataset_type)
  )[0]

  function getBadges(modelName: string) {
    const badges: string[] = []
    if (safestModel?.model === modelName) badges.push("🛡 Safest")
    if (fastestSafeModel?.model === modelName) badges.push("⚡ Fast Safe")
    if (balancedModel?.model === modelName) badges.push("⚖️ Balanced")
    return badges
  }

  /* ================= UI STATES ================= */

  if (loading) return <p className="p-8">Loading…</p>
  if (!doc) return <p className="p-8 text-red-500">Error loading</p>

  const selectedModelData =
    selectedModel === "ALL"
      ? null
      : sortedChartData.find((m) => m.model === selectedModel)

  /* ================= RENDER ================= */

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-10">

      {/* ===== SORT + MODEL SELECT ===== */}
      <div className="flex flex-wrap gap-6">

        <div>
          <label className="text-sm font-medium">Sort</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="border rounded-lg px-3 py-2 block"
          >
            <option value="balanced">⚖️ Balanced</option>
            <option value="latency">⚡ Fastest</option>
            <option value="confidence">🎯 Confidence</option>
            <option value="risk">🛡 Safest</option>
          </select>
        </div>

        <div>
          <label className="text-sm font-medium">View Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="border rounded-lg px-3 py-2 block"
          >
            <option value="ALL">All Models</option>
            {MODEL_ORDER.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </div>

      </div>

      {/* ===== CHARTS ===== */}
      {selectedModel === "ALL" && (
        <ChartSection data={sortedChartData as ChartItem[]} selectedModel="ALL" />
      )}

      {/* ===== MODEL CARDS ===== */}
      {(selectedModel === "ALL" ? sortedChartData : [selectedModelData]).map(
        (m) =>
          m && (
            <div key={m.model} className="border rounded-xl p-6">
              <h3 className="font-semibold">{m.model}</h3>

              <div className="flex gap-2 my-2">
                {getBadges(m.model).map((b) => (
                  <span key={b} className="text-xs bg-blue-100 px-2 py-1 rounded">
                    {b}
                  </span>
                ))}
              </div>

              <MetricRow label="Confidence" value={m.confidence_percent.toFixed(2) + "%"} />
              <MetricRow label="Latency" value={m.latency_ms.toFixed(3) + " ms"} />
              <MetricRow label="Risk Score" value={m.risk_score.toFixed(4)} />
            </div>
          )
      )}

      {/* ===== CONFUSION MATRICES ===== */}
      {selectedModel === "ALL" && (
        <div className="space-y-8">
          {confusionMatrices.map((m) => (
            <ConfusionMatrix key={m.model} data={m} />
          ))}
        </div>
      )}

      {selectedModel !== "ALL" && (
        <ConfusionMatrix
          data={confusionMatrices.find((m) => m.model === selectedModel)!}
        />
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
