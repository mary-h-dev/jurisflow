// middleware.ts
import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const PUBLIC_PATHS = ["/login", "/register"]

export function middleware(request: NextRequest) {
  const access  = request.cookies.get("access_token")?.value
  const refresh = request.cookies.get("refresh_token")?.value
  const hasSession = !!access || !!refresh
  const pathname = request.nextUrl.pathname

  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    if (hasSession) return NextResponse.redirect(new URL("/search", request.url))
    return NextResponse.next()
  }

  if (!hasSession) {
    return NextResponse.redirect(new URL("/login", request.url))
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
}