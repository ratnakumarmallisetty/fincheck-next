import type { ChartItem } from "./types"

export const METRIC_META: Record<
  keyof ChartItem,
  {
    label: string
    description: string
    higherIsBetter: boolean
  }
> = {
  model: {
    label: "Model",
    description: "",
    higherIsBetter: true,
  },

  confidence_percent: {
    label: "Confidence (%)",
    description:
      "Maximum softmax probability. Higher confidence indicates more certain predictions.",
    higherIsBetter: true,
  },

  latency_ms: {
    label: "Latency (ms)",
    description:
      "End-to-end inference time per image. Lower latency is better.",
    higherIsBetter: false,
  },

  entropy: {
    label: "Prediction Entropy",
    description:
      "Uncertainty of the output probability distribution. Lower is better.",
    higherIsBetter: false,
  },

  stability: {
    label: "Logit Stability",
    description:
      "Standard deviation of logits across classes. Lower values indicate more stable predictions.",
    higherIsBetter: false,
  },

  ram_delta_mb: {
    label: "RAM Usage (MB)",
    description:
      "Additional memory consumed during inference. Lower memory usage is better.",
    higherIsBetter: false,
  },
}
