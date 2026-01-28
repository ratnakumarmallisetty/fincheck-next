"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import EvaluationModeSelector from "../../../components/EvaluationModeSelector"
import { PREBUILT_DATASETS } from "@/lib/dataset"

type Mode = "SINGLE" | "DATASET"
type DatasetSource = "PREBUILT" | "CUSTOM"

export default function ConMatPage() {
  const router = useRouter()

  const [mode, setMode] = useState<Mode>("SINGLE")
  const [datasetSource, setDatasetSource] =
    useState<DatasetSource>("PREBUILT")

  const [selectedDataset, setSelectedDataset] =
    useState<string>("MNIST_100")

  const [selectedDigit, setSelectedDigit] = useState<number>(0)

  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function runInference() {
    setLoading(true)
    setError(null)

    const form = new FormData()
    let endpoint = ""

    if (mode === "SINGLE") {
      if (!file) {
        setError("Please upload an image")
        setLoading(false)
        return
      }
      form.append("image", file)
      form.append("expected_digit", selectedDigit.toString())
      endpoint = "/api/run"
    } else {
      endpoint = "/api/run-dataset"

      if (datasetSource === "CUSTOM") {
        if (!file) {
          setError("Please upload a ZIP file")
          setLoading(false)
          return
        }
        form.append("zip_file", file)
      } else {
        form.append("dataset_name", selectedDataset)
      }
    }

    const res = await fetch(endpoint, {
      method: "POST",
      body: form,
    })

    if (!res.ok) {
      setError("Inference failed. Please try again.")
      setLoading(false)
      return
    }

    const json = await res.json()

    const resultId = json.id || json.result_id

    if (!resultId) {
      setError("Invalid response from server")
      setLoading(false)
      return
    }

    router.push(`/results/${resultId}`)
  }

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-8">
      <h1 className="text-3xl font-bold">Run Inference</h1>

      <EvaluationModeSelector mode={mode} setMode={setMode} />

      {/* ---------- SINGLE IMAGE MODE ---------- */}
      {mode === "SINGLE" && (
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Expected Digit
            </label>
            <select
              value={selectedDigit}
              onChange={(e) =>
                setSelectedDigit(Number(e.target.value))
              }
              className="rounded-lg border border-gray-300 px-3 py-2 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => (
                <option key={digit} value={digit}>
                  {digit}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              Upload Image
            </label>
            <p className="text-sm text-gray-600">
              Upload a single handwritten digit image
            </p>
            <input
              type="file"
              accept="image/*"
              onChange={(e) =>
                setFile(e.target.files?.[0] || null)
              }
              className="block w-full text-sm text-gray-500 file:mr-4 file:rounded-lg file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
        </div>
      )}

      {/* ---------- DATASET MODE ---------- */}
      {mode === "DATASET" && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Evaluate models on a dataset for comparative analysis
          </p>

          <div className="flex gap-6">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={datasetSource === "PREBUILT"}
                onChange={() =>
                  setDatasetSource("PREBUILT")
                }
              />
              Prebuilt dataset
            </label>

            <label className="flex items-center gap-2">
              <input
                type="radio"
                checked={datasetSource === "CUSTOM"}
                onChange={() =>
                  setDatasetSource("CUSTOM")
                }
              />
              Custom ZIP upload
            </label>
          </div>

          {datasetSource === "PREBUILT" && (
            <select
              value={selectedDataset}
              onChange={(e) =>
                setSelectedDataset(e.target.value)
              }
              className="rounded-lg border px-3 py-2"
            >
              {PREBUILT_DATASETS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          )}

          {datasetSource === "CUSTOM" && (
            <input
              type="file"
              accept=".zip"
              onChange={(e) =>
                setFile(e.target.files?.[0] || null)
              }
            />
          )}
        </div>
      )}

      {error && (
        <p className="text-sm text-red-600">{error}</p>
      )}

      <button
        onClick={runInference}
        disabled={loading}
        className="rounded-lg bg-blue-600 px-5 py-2 text-white disabled:opacity-50"
      >
        {loading ? "Running…" : "Run Evaluation"}
      </button>
    </div>
  )
}