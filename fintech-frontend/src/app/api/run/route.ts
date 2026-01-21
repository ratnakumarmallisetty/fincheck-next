import { NextResponse } from "next/server";
import { connectMongo } from "@/lib/mongodb";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    // ✅ DO NOT re-wrap the file
    const formData = await req.formData();

    const file = formData.get("image");
    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "No image provided" },
        { status: 400 }
      );
    }

    const API_URL = process.env.INFERENCE_API_URL;
    if (!API_URL) {
      console.error("INFERENCE_API_URL not set");
      return NextResponse.json(
        { error: "Inference API not configured" },
        { status: 500 }
      );
    }

    // ✅ Forward multipart stream directly
    const r = await fetch(`${API_URL}/run`, {
      method: "POST",
      body: formData,
    });

    const text = await r.text();

    if (!r.ok) {
      console.error("Backend /run failed:", text);
      return NextResponse.json(
        { error: text },
        { status: 500 }
      );
    }

    const data = JSON.parse(text);

    const db = await connectMongo();
    const result = await db.collection("model_results").insertOne({
      data,
      meta: {
        evaluation_type: "SINGLE",
        source: "IMAGE_UPLOAD",
      },
      createdAt: new Date(),
    });

    return NextResponse.json({ id: result.insertedId });
  } catch (err) {
    console.error("run API error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
