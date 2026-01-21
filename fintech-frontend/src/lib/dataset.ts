export type PrebuiltDataset = {
  id: string
  label: string
  description: string
}

export const PREBUILT_DATASETS: PrebuiltDataset[] = [
  /* ---------- CLEAN MNIST ---------- */
  {
    id: "MNIST_100",
    label: "MNIST (100 samples)",
    description: "Quick evaluation using a small clean MNIST subset",
  },
  {
    id: "MNIST_500",
    label: "MNIST (500 samples)",
    description: "Balanced evaluation with a moderate clean MNIST subset",
  },
  {
    id: "MNIST_FULL",
    label: "MNIST (Full test set)",
    description: "Complete MNIST test set (10,000 images) for full benchmarking",
  },

  /* ---------- NOISY MNIST ---------- */
  {
    id: "MNIST_NOISY_100",
    label: "MNIST Noisy (100 samples)",
    description: "MNIST subset with additive Gaussian noise for robustness testing",
  },
  {
    id: "MNIST_NOISY_500",
    label: "MNIST Noisy (500 samples)",
    description: "Larger noisy MNIST subset to evaluate robustness at scale",
  },

  /* ---------- BLURRED MNIST ---------- */
  {
    id: "MNIST_BLUR_100",
    label: "MNIST Blurred (100 samples)",
    description: "MNIST subset with Gaussian blur to test spatial robustness",
  },
  {
    id: "MNIST_BLUR_500",
    label: "MNIST Blurred (500 samples)",
    description: "Moderately sized blurred MNIST subset for robustness analysis",
  },

  /* ---------- NOISY + BLURRED MNIST ---------- */
  {
    id: "MNIST_NOISY_BLUR_100",
    label: "MNIST Noisy + Blurred (100 samples)",
    description: "MNIST subset with both noise and blur for extreme robustness testing",
  },
  {
    id: "MNIST_NOISY_BLUR_500",
    label: "MNIST Noisy + Blurred (500 samples)",
    description: "Larger noisy+blurred subset for stress-testing model robustness",
  },
]