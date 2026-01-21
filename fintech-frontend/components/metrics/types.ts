export type ChartItem = {
  model: string

  // --- core metrics (always present) ---
  confidence_percent: number
  latency_ms: number
  entropy: number
  stability: number
  ram_delta_mb: number

  // --- optional stats for noisy datasets ---
  confidence_std?: number
  latency_std?: number
  entropy_std?: number
  stability_std?: number
}