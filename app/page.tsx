import Header from "@/components/Header"

export default function Home() {
  return (
    <>
      <Header />

      <main className="mx-auto max-w-5xl px-6 py-12 space-y-12">
        {/* Hero / Introduction */}
        <section className="space-y-4">
          <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
            Efficient CNN Model Compression Analysis
          </h1>

          <p className="text-gray-600 leading-relaxed max-w-3xl">
            Convolutional Neural Networks (CNNs) achieve state-of-the-art
            performance in image recognition tasks. However, their high
            computational and memory requirements often make deployment
            challenging on resource-constrained devices.
          </p>
        </section>

        {/* Project Description */}
        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">
            Project Overview
          </h2>

          <p className="text-gray-600 leading-relaxed">
            Several model compression techniques—such as <span className="font-medium text-gray-800">pruning</span>,{" "}
            <span className="font-medium text-gray-800">quantization</span>,{" "}
            <span className="font-medium text-gray-800">low-rank factorization</span>,{" "}
            <span className="font-medium text-gray-800">knowledge distillation</span>, and{" "}
            <span className="font-medium text-gray-800">weight sharing</span>—have been
            proposed to reduce model complexity. Despite their popularity, there
            is a lack of systematic comparison under consistent experimental
            settings.
          </p>

          <p className="text-gray-600 leading-relaxed">
            This project aims to evaluate and compare these five compression
            techniques using a standard CNN trained on the <span className="font-medium text-gray-800">MNIST dataset</span>.
            The study focuses on understanding the trade-offs between efficiency
            and reliability.
          </p>
        </section>

        {/* Evaluation Metrics */}
        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">
            Evaluation Criteria
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              "Model accuracy on clean inputs",
              "Inference time performance",
              "Model size and memory footprint",
              "Robustness against noisy inputs",
            ].map((item) => (
              <div
                key={item}
                className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700"
              >
                {item}
              </div>
            ))}
          </div>

          <p className="text-gray-600 leading-relaxed">
            In addition to clean datasets, noise-added datasets are used to
            evaluate robustness. Performance is assessed based on both inference
            time and accuracy under varying conditions.
          </p>
        </section>

        {/* Conclusion */}
        <section className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-900">
            Objective
          </h2>

          <p className="text-gray-600 leading-relaxed max-w-3xl">
            The study seeks to provide insights into the trade-offs between
            computational efficiency and predictive reliability, offering
            practical guidance for selecting suitable compression methods when
            deploying CNNs in resource-constrained environments.
          </p>
        </section>

        {/* Team Section */}
        <section className="space-y-6">
          <h2 className="text-xl font-semibold text-gray-900">
            Team Members
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { name: "Sai Vikas", id: "CB.EN.U4CSE22363" },
              { name: "Albert", id: "CB.EN.U4CSE22505" },
              { name: "Rathna", id: "CB.EN.U4CSE22526" },
              { name: "Mukesh", id: "CB.EN.U4CSE22531" },
            ].map((member) => (
              <div
                key={member.id}
                className="rounded-lg border border-gray-200 px-4 py-3"
              >
                <p className="font-medium text-gray-900">
                  {member.name}
                </p>
                <p className="text-sm text-gray-600">
                  {member.id}
                </p>
              </div>
            ))}
          </div>
        </section>
      </main>
    </>
  )
}
