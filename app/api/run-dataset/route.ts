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

    const text = await r.text()

    let data: any
    try {
      data = JSON.parse(text)
    } catch {
      console.error("Non-JSON backend response:", text)
      return NextResponse.json(
        { error: "Invalid response from inference backend" },
        { status: 500 }
      )
    }

    // ❌ Backend-level error (logical error, not crash)
    if (data?.error) {
      console.error("Backend dataset error:", data.error)
      return NextResponse.json(
        { error: data.error },
        { status: 400 } // important: NOT 500
      )
    }

    // ❌ Unexpected shape
    if (!data.models) {
      console.error("Backend returned unexpected payload:", data)
      return NextResponse.json(
        { error: "Malformed inference result from backend" },
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
