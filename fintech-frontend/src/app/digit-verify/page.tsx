"use client"

import { useState } from "react"

export default function DigitVerifyPage() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  async function verify() {
    if (!file) return

    const fd = new FormData()
    fd.append("image", file)

    setLoading(true)
    setResult(null)

    const res = await fetch("http://localhost:8000/verify-digit-only", {
      method: "POST",
      body: fd,
    })

    const data = await res.json()
    setResult(data)
    setLoading(false)
  }

  return (
    <main className="p-10 max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-semibold">
        Cheque Digit Validation
      </h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      <button
        onClick={verify}
        disabled={!file || loading}
        className="bg-black text-white px-4 py-2 rounded"
      >
        {loading ? "Checking..." : "Verify Image"}
      </button>

      {result && (
        <div className="bg-gray-100 p-4 rounded space-y-3">
          <p><b>Verdict:</b> {result.verdict}</p>
          <p><b>Detected Digits:</b> {result.digits}</p>

          <div className="space-y-2">
            {result.analysis.map((a: any) => (
              <div key={a.position} className="border p-2 rounded">
                <p>Position {a.position}</p>
                <p>Status: <b>{a.status}</b></p>
                <p>Predicted: {a.predicted}</p>
                <p>Confidence: {a.confidence}%</p>
                <p>Possible values: {a.possible_values.join(", ")}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  )
}
