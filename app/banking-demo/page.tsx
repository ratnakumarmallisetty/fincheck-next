"use client"

import { useState } from "react"

type Result = {
  confidence: number
  latency_ms: number
  ram_mb: number
}

export default function BankingDemoPage() {
  const [image, setImage] = useState<File | null>(null)
  const [results, setResults] = useState<Record<string, Result> | null>(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState("cheque")

  async function runInference() {
    if (!image) return

    setLoading(true)
    const formData = new FormData()
    formData.append("image", image)

    const res = await fetch("http://localhost:8000/run", {
      method: "POST",
      body: formData,
    })

    const data = await res.json()
    setResults(data)
    setLoading(false)
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-12 space-y-8">
      <h1 className="text-3xl font-semibold">
        Banking ML Compression Demo
      </h1>

      {/* Mode Selector */}
      <div className="flex gap-3">
        {[
          { id: "cheque", label: "Cheque Verification" },
          { id: "atm", label: "ATM Edge Inference" },
          { id: "mobile", label: "Mobile OCR Filter" },
        ].map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-4 py-2 rounded-md border text-sm ${
              mode === m.id
                ? "bg-black text-white"
                : "bg-white text-gray-700"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Mode Description */}
      <p className="text-gray-600 max-w-3xl">
        {mode === "cheque" &&
          "Simulates handwritten digit verification on bank cheques under noisy scanning conditions."}
        {mode === "atm" &&
          "Benchmarks compressed CNN models for low-latency inference on ATM-grade hardware."}
        {mode === "mobile" &&
          "Evaluates lightweight OCR filtering for mobile banking apps to reduce server load."}
      </p>

      {/* Upload */}
      <div className="space-y-4">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files?.[0] || null)}
        />

        <button
          onClick={runInference}
          disabled={!image || loading}
          className="rounded-md bg-blue-600 px-4 py-2 text-white text-sm disabled:opacity-50"
        >
          {loading ? "Running Models..." : "Run Inference"}
        </button>
      </div>

      {/* Results */}
      {results && (
        <div className="overflow-x-auto">
          <table className="w-full border border-gray-200 text-sm">
            <thead className="bg-gray-100">
              <tr>
                <th className="p-3 text-left">Model</th>
                <th className="p-3 text-left">Confidence (%)</th>
                <th className="p-3 text-left">Latency (ms)</th>
                <th className="p-3 text-left">RAM Δ (MB)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results).map(([name, r]) => (
                <tr key={name} className="border-t">
                  <td className="p-3 font-medium">{name}</td>
                  <td className="p-3">{r.confidence.toFixed(2)}</td>
                  <td className="p-3">{r.latency_ms}</td>
                  <td className="p-3">{r.ram_mb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  )
}
