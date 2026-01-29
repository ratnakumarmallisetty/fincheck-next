"use client"

import { useState } from "react"

export default function DigitVerifyPage() {

  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  function statusColor(status?: string) {
    if (!status) return "text-gray-500"
    if (status === "VALID") return "text-green-600"
    if (status === "AMBIGUOUS") return "text-yellow-600"
    return "text-red-600"
  }

  async function verify() {

    if (!file) return

    const fd = new FormData()
    fd.append("image", file)

    setLoading(true)
    setResult(null)

    const res = await fetch(
      "http://localhost:8000/verify-digit-only",
      { method: "POST", body: fd }
    )

    const data = await res.json()
    setResult(data)
    setLoading(false)
  }

  function imgSrc(b64?: string) {
    if (!b64) return ""
    return `data:image/png;base64,${b64}`
  }

  return (
    <main className="p-10 max-w-4xl mx-auto space-y-6">

      <h1 className="text-2xl font-semibold">
        Cheque Digit Validation
      </h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e) =>
          setFile(e.target.files?.[0] || null)
        }
      />

      <button
        onClick={verify}
        disabled={!file || loading}
        className="bg-black text-white px-4 py-2 rounded"
      >
        {loading ? "Checking..." : "Verify Image"}
      </button>

      {result && (
        <div className="space-y-6">

          {/* ===== MAIN RESULT ===== */}
          <div className="bg-gray-100 p-4 rounded">
            <p>
              <b>Verdict:</b>{" "}
              <span className={statusColor(result.verdict)}>
                {result.verdict}
              </span>
            </p>

            <p>
              <b>Digits:</b> {result.digits || "???"}
            </p>
          </div>

          {/* ===== PREVIEW IMAGES ===== */}
          {result.preview && (
            <div className="grid grid-cols-3 gap-4">

              <div className="border p-2 rounded text-center">
                <p className="font-medium mb-2">Original</p>
                <img src={imgSrc(result.preview.original)} />
              </div>

              <div className="border p-2 rounded text-center">
                <p className="font-medium mb-2">Cropped</p>
                <img src={imgSrc(result.preview.cropped)} />
              </div>

              <div className="border p-2 rounded text-center">
                <p className="font-medium mb-2">Normalized 28×28</p>
                <img src={imgSrc(result.preview.normalized)} />
              </div>

            </div>
          )}

          {/* ===== PER DIGIT ===== */}
          {result.analysis?.map((a: any) => (
            <div key={a.position} className="border p-3 rounded">

              <p className="font-semibold">
                Position {a.position}
              </p>

              <p className={statusColor(a.status)}>
                Status: {a.status}
              </p>

              <p>
                Confidence: {a.confidence}%
              </p>

              {a.possible_values?.length > 0 && (
                <p className="text-sm text-gray-600">
                  Possible: {a.possible_values.join(", ")}
                </p>
              )}

            </div>
          ))}

        </div>
      )}

    </main>
  )
}
