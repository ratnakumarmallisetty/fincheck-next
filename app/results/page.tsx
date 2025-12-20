import Link from "next/link"
import connectDB from "@/lib/mongodb"

export const dynamic = "force-dynamic"

type ResultDoc = {
  _id: any
  createdAt: Date
  data: Record<string, any>
  meta?: {
    evaluation_type?: "SINGLE" | "DATASET"
    source?: "PREBUILT" | "CUSTOM" | "IMAGE_UPLOAD"
    dataset_type?: string
    num_images?: number
  }
}

/* ---------- META NORMALIZATION ---------- */
function normalizeMeta(meta?: ResultDoc["meta"]) {
  return {
    evaluation_type: meta?.evaluation_type ?? "SINGLE",
    source: meta?.source ?? "IMAGE_UPLOAD",
    dataset_type: meta?.dataset_type,
    num_images: meta?.num_images,
  }
}

export default async function ResultsPage() {
  const db = await connectDB()

  const docs = (await db
    .collection("model_results")
    .find()
    .sort({ createdAt: -1 })
    .toArray()) as ResultDoc[]

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-8">
      <h1 className="text-3xl font-bold">
        Inference History
      </h1>

      {docs.map((doc) => {
        const meta = normalizeMeta(doc.meta)

        return (
          <Link
            key={doc._id.toString()}
            href={`/results/${doc._id}`}
            className="block rounded-xl border bg-white p-5 hover:shadow-md"
          >
            <p className="text-sm text-gray-500">
              {new Date(doc.createdAt).toLocaleString()}
            </p>

            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full bg-blue-100 px-2 py-0.5 text-blue-700">
                {meta.evaluation_type === "SINGLE"
                  ? "Single Image"
                  : "Dataset"}
              </span>

              <span className="rounded-full bg-purple-100 px-2 py-0.5 text-purple-700">
                {meta.source}
              </span>

              {meta.dataset_type && (
                <span className="rounded-full bg-green-100 px-2 py-0.5 text-green-700">
                  {meta.dataset_type}
                </span>
              )}

              {meta.num_images && (
                <span className="rounded-full bg-gray-200 px-2 py-0.5 text-gray-700">
                  {meta.num_images} images
                </span>
              )}
            </div>
          </Link>
        )
      })}

      {docs.length === 0 && (
        <p className="text-sm text-gray-500">
          No inference results found yet.
        </p>
      )}
    </div>
  )
}
