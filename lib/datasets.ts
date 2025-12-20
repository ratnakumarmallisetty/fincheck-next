export type PrebuiltDataset = {
  id: string
  label: string
  description: string
}

export const PREBUILT_DATASETS: PrebuiltDataset[] = [
  {
    id: "MNIST_100",
    label: "MNIST (100 samples)",
    description: "Quick evaluation using a small MNIST subset",
  },
  {
    id: "MNIST_500",
    label: "MNIST (500 samples)",
    description: "Balanced evaluation with moderate dataset size",
  },
  {
    id: "MNIST_TEST",
    label: "MNIST (Full test set)",
    description: "Complete MNIST test set for full benchmarking",
  },
  {
    id: "MNIST_NOISY_100",
    label: "MNIST Noisy (100 samples)",
    description: "MNIST subset with added noise for robustness testing",
  },
]
