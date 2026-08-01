"use client";

import * as Sentry from "@sentry/nextjs";
import { useEffect } from "react";

export default function GlobalError({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <body>
        <main>
          <h1>The review frontend encountered an error.</h1>
          <p>The local MCP process and saved plans are separate from this interface.</p>
          <button type="button" onClick={unstable_retry}>
            Try again
          </button>
        </main>
      </body>
    </html>
  );
}
