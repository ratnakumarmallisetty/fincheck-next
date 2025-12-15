import { getServerSession } from "next-auth"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import SignOutButton from "./sign-out-button"

export default async function Header() {
  const session = await getServerSession(authOptions)

  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur border-b border-gray-200">
      {/* Top bar */}
      <div className="mx-auto max-w-6xl px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Brand / Logout */}
          {session ? (
            <SignOutButton />
          ) : (
            <div className="text-sm font-semibold tracking-tight text-gray-950">
              Fincheck
            </div>
          )}

          {/* Auth actions */}
          {!session && (
            <div className="flex items-center gap-3 text-sm">
              <a
                href="/sign-in"
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                Login
              </a>
              <a
                href="/sign-up"
                className="rounded-md bg-gray-900 px-3 py-1.5 text-white hover:bg-gray-800 transition-colors"
              >
                Sign up
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="bg-gray-50/80 border-t border-gray-200">
        <div className="mx-auto max-w-6xl px-6">
          <ul className="flex gap-6 text-sm text-gray-600">
            {[
              { href: "/image-loader", label: "Image Loader" },
              { href: "/results", label: "Results" },
              { href: "/predictions", label: "Predictions" },
            ].map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  className="relative block py-2 transition-colors hover:text-gray-900
                             after:absolute after:inset-x-0 after:-bottom-px after:h-px
                             after:scale-x-0 after:bg-gray-900 after:transition-transform
                             hover:after:scale-x-100"
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    </header>
  )
}
