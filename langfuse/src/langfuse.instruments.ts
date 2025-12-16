import { NodeSDK } from '@opentelemetry/sdk-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';

export const langfuseSpanProcessor = new LangfuseSpanProcessor({
  shouldExportSpan: ({ otelSpan }) => otelSpan.instrumentationScope.name === 'langfuse-sdk',
  flushAt: 5,
  flushInterval: 1,
});

const sdk = new NodeSDK({
  spanProcessors: [langfuseSpanProcessor],
});

sdk.start();
