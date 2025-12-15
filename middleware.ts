import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"
import { getToken } from "next-auth/jwt"

export async function middleware(req: NextRequest) {
  const token = await getToken({
    req,
    secret: process.env.NEXTAUTH_SECRET,
  })

  const pathname = req.nextUrl.pathname

  // ✅ Public routes (ALLOWED without login)
  if (
    pathname === "/" ||
    pathname.startsWith("/sign-in") ||
    pathname.startsWith("/sign-up") ||
    pathname.startsWith("/api/auth") ||
    pathname.startsWith("/_next") ||
    pathname === "/favicon.ico"
  ) {
    return NextResponse.next()
  }

  // ❌ Everything else requires login
  if (!token) {
    return NextResponse.redirect(
      new URL("/sign-in", req.url)
    )
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/:path*"],
}
