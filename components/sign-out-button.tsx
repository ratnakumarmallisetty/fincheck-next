"use client"

import { signOut } from "next-auth/react"

export default function SignOutButton() {
  return (
    <button
      onClick={() => signOut()}
      className="text-sm font-semibold tracking-tight text-gray-950 hover:text-gray-700 transition-colors"
      title="Sign out"
    >
      Fincheck
    </button>
  )
}
