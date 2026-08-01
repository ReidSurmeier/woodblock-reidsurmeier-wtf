import * as Sentry from "@sentry/nextjs";

export function register() {
  const publicDsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
  const serverDsn = process.env.SENTRY_DSN;
  const isNodeRuntime = process.env.NEXT_RUNTIME === "nodejs";
  const dsn = isNodeRuntime ? serverDsn || publicDsn : publicDsn;

  Sentry.init({
    dsn: dsn || "",
    tracesSampleRate: 0.1,
    enabled: Boolean(dsn),
  });
}

export const onRequestError = Sentry.captureRequestError;
