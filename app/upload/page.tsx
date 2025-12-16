"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  async function submit() {
    if (!file) return
    setLoading(true)

    try {
      const fd = new FormData()
      fd.append("image", file)

      const res = await fetch("/api/upload", {
        method: "POST",
        body: fd,
      })

      if (!res.ok) {
        const text = await res.text()
        console.error("Upload failed:", text)
        alert("Upload failed. Check backend logs.")
        return
      }

      const data = await res.json()
      router.push(`/results/${data.id}`)
    } catch (err) {
      console.error(err)
      alert("Something went wrong.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient from-gray-50 to-gray-100 flex items-center justify-center">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg space-y-6">
        <h1 className="text-3xl font-bold text-gray-900 text-center">
          Upload Image
        </h1>

        <p className="text-sm text-gray-500 text-center">
          Upload a handwritten digit image to run all models
        </p>

        <label className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 cursor-pointer hover:border-black transition">
          <input
            type="file"
            className="hidden"
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
          <span className="text-sm text-gray-600">
            {file ? file.name : "Click to select an image"}
          </span>
        </label>

        <button
          onClick={submit}
          disabled={loading || !file}
          className="w-full rounded-xl bg-black py-3 text-white font-semibold
                     hover:bg-gray-800 disabled:opacity-50 transition"
        >
          {loading ? "Running Models..." : "Run Inference"}
        </button>
      </div>
    </div>
  )
}
