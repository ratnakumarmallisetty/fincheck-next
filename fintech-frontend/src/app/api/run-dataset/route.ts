import { NextResponse } from "next/server";
import { connectMongo } from "@/lib/mongodb";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    const incomingForm = await req.formData();

    const datasetName = incomingForm.get("dataset_name");
    const zipFile = incomingForm.get("zip_file"); 

    if (!datasetName && !(zipFile instanceof File)) {
      return NextResponse.json(
        { error: "No dataset_name or zip_file provided" },
        { status: 400 }
      );
    }

    const API_URL = process.env.INFERENCE_API_URL;
    if (!API_URL) {
      return NextResponse.json(
        { error: "Inference API not configured" },
        { status: 500 }
      );
    }

    // ---------- Rebuild FormData ----------
    const forwardForm = new FormData();

    if (datasetName) {
      forwardForm.append(
        "dataset_name",
        datasetName.toString()
      );
    }

    if (zipFile instanceof File) {
      forwardForm.append("zip_file", zipFile);
    }

    // ---------- Forward to FastAPI ----------
    const r = await fetch(
      `${API_URL}/run-dataset`,
      { method: "POST", body: forwardForm }
    );

    const text = await r.text();

    if (!r.ok) {
      console.error("Backend /run-dataset failed:", text);
      return NextResponse.json(
        { error: text },
        { status: 500 }
      );
    }

    const data = JSON.parse(text);

    if (!data.models) {
      return NextResponse.json(
        { error: "Malformed backend response" },
        { status: 500 }
      );
    }

    // ---------- Store in Mongo ----------
    const db = await connectMongo();
    const result = await db
      .collection("model_results")
      .insertOne({
        data: data.models,
        meta: {
          evaluation_type: "DATASET",
          source: zipFile ? "CUSTOM_ZIP" : "PREBUILT",
          dataset_type: data.dataset_type,
          num_images: data.num_images,
        },
        createdAt: new Date(),
      });

    return NextResponse.json({
      id: result.insertedId.toString(),
    });

  } catch (err) {
    console.error("run-dataset API error:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
