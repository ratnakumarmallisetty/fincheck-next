import { NextResponse } from "next/server"
import connectDB from "@/lib/mongodb"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

export async function POST(req: Request) {
  try {
    const formData = await req.formData()

    const API_URL = process.env.INFERENCE_API_URL
    if (!API_URL) {
      return NextResponse.json(
        { error: "INFERENCE_API_URL not set" },
        { status: 500 }
      )
    }

    const r = await fetch(`${API_URL}/run-dataset`, {
      method: "POST",
      body: formData,
    })

    if (!r.ok) {
      const text = await r.text()
      console.error("Backend /run-dataset failed:", text)
      return NextResponse.json(
        { error: "Backend dataset inference failed" },
        { status: 500 }
      )
    }

    const data = await r.json()

    if (!data || !data.models) {
      return NextResponse.json(
        { error: "Invalid backend response" },
        { status: 500 }
      )
    }

    const datasetName = formData.get("dataset_name")
    const source = datasetName ? "PREBUILT" : "CUSTOM"

    const db = await connectDB()
    const result = await db.collection("model_results").insertOne({
      data: data.models,
      meta: {
        evaluation_type: "DATASET",
        source,
        dataset_type: data.dataset_type,
        num_images: data.num_images,
      },
      createdAt: new Date(),
    })

    return NextResponse.json({ id: result.insertedId })
  } catch (err) {
    console.error("run-dataset API error:", err)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
