import { NextResponse } from "next/server"
import { connectMongo } from "@/lib/mongodb"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

export async function POST(req: Request) {
  const formData = await req.formData()

  const r = await fetch(
    `${process.env.INFERENCE_API_URL}/run-dataset`,
    { method: "POST", body: formData }
  )

  const data = await r.json()

  if (!data.models) {
    return NextResponse.json(
      { error: "Malformed backend response" },
      { status: 500 }
    )
  }

  const db = await connectMongo()

  const result = await db.collection("model_results").insertOne({
    data: data.models,
    meta: {
      evaluation_type: "DATASET",
      source: "PREBUILT",
      dataset_type: data.dataset_type,
      num_images: data.num_images,
    },
    createdAt: new Date(),
  })

  return NextResponse.json({ id: result.insertedId })
}
