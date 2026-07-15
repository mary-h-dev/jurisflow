import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const PUBLIC_PATHS  = ["/login", "/register"]
const ONBOARDING    = ["/onboarding"]

export function middleware(request: NextRequest) {
  const token    = request.cookies.get("access_token")?.value
  const pathname = request.nextUrl.pathname


  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    if (token) return NextResponse.redirect(new URL("/search", request.url))
    return NextResponse.next()
  }


  if (!token) {
    return NextResponse.redirect(new URL("/login", request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}