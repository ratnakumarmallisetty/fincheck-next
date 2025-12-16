import { NextResponse } from "next/server"
import connectDB from "@/lib/mongodb"

export async function POST(req: Request) {
  try {
    const form = await req.formData()
    const file = form.get("image")

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "No image provided" },
        { status: 400 }
      )
    }

    // 🔥 Convert Web File → Buffer
    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    const fd = new FormData()
    fd.append("image", new Blob([buffer]), file.name)

    const r = await fetch("http://127.0.0.1:8000/run", {
      method: "POST",
      body: fd,
    })

    if (!r.ok) {
      const text = await r.text()
      console.error("Backend /run failed:", text)
      return NextResponse.json(
        { error: "Backend inference failed" },
        { status: 500 }
      )
    }

    const data = await r.json()

    const db = await connectDB()
    const result = await db.collection("model_results").insertOne({
      data,
      createdAt: new Date(),
    })

    return NextResponse.json({ id: result.insertedId })
  } catch (err) {
    console.error("Upload API error:", err)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
