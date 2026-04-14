import { LangfuseSpanProcessor } from '@langfuse/otel';

export const langfuseSpanProcessor = new LangfuseSpanProcessor({
  flushAt: 1,
  flushInterval: 1,
  exportMode: 'immediate',
});
