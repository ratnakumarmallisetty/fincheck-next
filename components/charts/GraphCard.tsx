"use client"

export default function GraphCard({
  title,
  description,
  bestLabel,
  children,
}: {
  title: string
  description?: string
  bestLabel?: string
  children: React.ReactNode
}) {
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm space-y-2">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700">
            {title}
          </h3>
          {description && (
            <p className="mt-1 text-xs text-gray-500 max-w-xl">
              {description}
            </p>
          )}
        </div>

        {bestLabel && (
          <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-700">
            🏆 Best: {bestLabel}
          </span>
        )}
      </div>

      {children}
    </div>
  )
}
