import { NextResponse } from "next/server";
import { connectMongo } from "@/lib/mongodb";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    const incomingForm = await req.formData();

    // ---------- Extract fields ----------
    const image = incomingForm.get("image");
    const expectedDigit = incomingForm.get("expected_digit");

    if (!(image instanceof File)) {
      return NextResponse.json(
        { error: "No image provided" },
        { status: 400 }
      );
    }

    if (expectedDigit === null) {
      return NextResponse.json(
        { error: "Expected digit not provided" },
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

    // ---------- Rebuild FormData for backend ----------
    const forwardForm = new FormData();
    forwardForm.append("image", image);
    forwardForm.append(
      "expected_digit",
      expectedDigit.toString()
    );

    // ---------- Forward to FastAPI ----------
    const r = await fetch(`${API_URL}/run`, {
      method: "POST",
      body: forwardForm,
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

    // ---------- Store in Mongo ----------
    const db = await connectMongo();
    const result = await db
      .collection("model_results")
      .insertOne({
        data,
        meta: {
          evaluation_type: "SINGLE",
          source: "IMAGE_UPLOAD",
          expected_digit: Number(expectedDigit),
        },
        createdAt: new Date(),
      });

    return NextResponse.json({
      id: result.insertedId.toString(),
    });

  } catch (err) {
    console.error("run API error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
