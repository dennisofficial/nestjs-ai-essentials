import { isDefined } from 'class-validator';
import { LangfuseObservation } from '@langfuse/tracing';
import { randomBytes } from 'crypto';
import { SpanContext } from '@opentelemetry/api';

export const safeModelParams = (params: Record<string, any>): Record<string, string | number> => {
  return Object.fromEntries(
    Object.entries(params)
      .map(([k, v]) => {
        if (typeof v === 'string' || typeof v === 'number') {
          return [k, v];
        }
      })
      .filter(isDefined),
  );
};

/**
 * Generate a random trace ID (32-character lowercase hexadecimal string)
 */
export const generateTraceId = (): string => {
  return randomBytes(16).toString('hex');
};

/**
 * Generate a random span ID (16-character lowercase hexadecimal string)
 */
export const generateSpanId = (): string => {
  return randomBytes(8).toString('hex');
};

export const parseParentContext = (observer: LangfuseObservation | undefined): SpanContext => {
  if (observer) {
    return observer.otelSpan.spanContext();
  }
  return {
    traceId: generateTraceId(),
    spanId: generateSpanId(),
    traceFlags: 1,
  };
};

export const cleanStackTrace = (stack: string | undefined): string | undefined => {
  if (!stack) return undefined;

  const lines = stack.split('\n');
  const errorMessage = lines[0];

  // Parse stack frames
  const frames = lines
    .slice(1)
    .map((line) => {
      // Clean up the line
      line = line
        .trim()
        .replace(/^at\s+/, '')
        .replace(process.cwd(), '');

      // Skip empty lines
      if (!line) return null;

      // Parse the stack frame: "functionName (location)" or just "functionName"
      const match = line.match(/^(.+?)\s+\((.+)\)$/);

      if (match) {
        const functionName = match[1].trim();
        const location = match[2].trim();

        // De-emphasize node_modules and node internals
        const isNodeModules = location.includes('/node_modules/') || location.startsWith('node:');

        return {
          functionName: isNodeModules ? functionName : `**${functionName}**`,
          location: isNodeModules ? `__${location}__` : `\`${location}\``,
        };
      }

      // If no parentheses, just use the whole line as function name
      return {
        functionName: `**${line}**`,
        location: '',
      };
    })
    .filter((frame) => frame !== null);

  // Build the table
  const table = [
    '| Function | Location |',
    '|----------|----------|',
    ...frames.map((frame) => `| ${frame.functionName} | ${frame.location} |`),
  ].join('\n');

  return `## ${errorMessage}\n\n${table}`;
};
