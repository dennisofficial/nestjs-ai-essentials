import { ROOT_CONTEXT, context, trace } from '@opentelemetry/api';
import {
  type LangfuseSpanAttributes,
  propagateAttributes,
  startObservation,
} from '@langfuse/tracing';

/**
 * A minimal, serializable pointer back to a triggering chat turn's trace. Used to LINK detached
 * background work (session turns) to the chat turn that spawned it — without nesting under it
 * (`sessionId` is trace-scoped, so detached work gets its own session-grouped trace instead).
 */
export interface ChatTracePointer {
  traceId: string;
  spanId: string;
}

/**
 * Best-effort capture of the currently-active OTel span as a serializable pointer. Returns
 * `undefined` when no span is active — the link is a bonus, never load-bearing. Cheap: it reads the
 * ambient OTel context and creates nothing.
 */
export function captureParentChatTrace(): ChatTracePointer | undefined {
  const ctx = trace.getActiveSpan()?.spanContext();
  return ctx ? { traceId: ctx.traceId, spanId: ctx.spanId } : undefined;
}

/** Trace-level metadata must be `Record<string,string>` — stringify non-string values. */
function stringifyMetadata(
  metadata?: Record<string, unknown>,
): Record<string, string> | undefined {
  if (!metadata) return undefined;
  const entries = Object.entries(metadata).filter(([, v]) => v !== undefined);
  if (entries.length === 0) return undefined;
  return Object.fromEntries(
    entries.map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)]),
  );
}

export interface TraceSessionTurnOptions {
  /** Observation + trace name (e.g. `session.turn:claude:execute`). */
  name: string;
  /** Groups this trace into a Langfuse session timeline — pass the harness session id. */
  sessionId: string;
  /** The turn's input message (the opening task / reply). */
  input?: unknown;
  /** Rich structured metadata kept on the OBSERVATION (objects allowed here). */
  metadata?: Record<string, unknown>;
  /** Trace-level tags. */
  tags?: string[];
}

/**
 * Run `fn` inside a single, self-rooted Langfuse observation — one standalone trace per call,
 * grouped by `sessionId`. Captures `fn`'s resolved value as the observation output; records ERROR
 * level + status message on throw. Exactly one observation per call (no child fan-out).
 *
 * Self-rooted: created under a cleared OTel context (`ROOT_CONTEXT`) with NO `parentSpanContext`, so
 * it never inherits an ambient parent nor fabricates a phantom one — the decided topology is one
 * standalone trace per session turn. Callers gate on their own `tracingEnabled`; this helper assumes
 * tracing is on (when no span processor is registered, `startObservation` is itself a cheap no-op).
 */
export async function traceSessionTurn<T>(
  fn: () => Promise<T>,
  opts: TraceSessionTurnOptions,
): Promise<T> {
  const { name, sessionId, input, metadata, tags } = opts;
  const observation = context.with(ROOT_CONTEXT, () =>
    propagateAttributes(
      {
        traceName: name,
        sessionId,
        tags,
        metadata: stringifyMetadata(metadata),
      },
      () =>
        startObservation(
          name,
          { input, metadata } as LangfuseSpanAttributes,
          { asType: 'span' },
        ),
    ),
  );
  try {
    const result = await fn();
    observation.update({ output: result } as LangfuseSpanAttributes);
    return result;
  } catch (e) {
    observation.update({
      level: 'ERROR',
      statusMessage: e instanceof Error ? (e.stack ?? e.message) : String(e),
    });
    throw e;
  } finally {
    observation.end();
  }
}
