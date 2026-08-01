import { NextRequest, NextResponse } from "next/server";

export function proxy(request: NextRequest) {
  const hostname = request.headers.get("host") ?? "";

  // color.reidsurmeier.wtf: serve color separator at root
  if (
    hostname.startsWith("color.") &&
    request.nextUrl.pathname === "/"
  ) {
    return NextResponse.rewrite(new URL("/color-separator", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/"],
};
