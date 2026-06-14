import { ROOT_CONTEXT, context, trace } from '@opentelemetry/api';
import {
  type LangfuseGenerationAttributes,
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

/**
 * Usage details to attach to a generation observation. Mirrors the Langfuse generation shape:
 * - `usageDetails` — token counts by tier (input, output, cache_read_input_tokens, etc.).
 * - `costDetails`  — cost breakdown; at minimum `{ total: costUsd }` when the SDK returns cost.
 * - `model`        — the real model id used for the run (for server-side cost inference when
 *   `costDetails` is absent, e.g. Codex / OpenAI).
 */
export interface IGenerationUsage {
  usageDetails?: Record<string, number>;
  costDetails?: Record<string, number>;
  model?: string;
}

export interface TraceSessionTurnOptions<T = unknown> {
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
  /**
   * Observation kind. Defaults to `'span'` to preserve the original behavior for existing callers
   * (rs-crm-app, cubix-infra, mls-studio). Pass `'generation'` to record token/cost usage and have
   * Langfuse render it in its cost/usage views.
   */
  asType?: 'span' | 'generation';
  /**
   * Optional callback to extract token/cost usage from the resolved result. Only applied when
   * `asType` is `'generation'`; ignored for spans. Called on success only; the returned object is
   * forwarded to the Langfuse generation as `usageDetails`, `costDetails`, and `model`. When absent
   * (or returns `undefined`) usage is not recorded.
   */
  usage?: (result: T) => IGenerationUsage | undefined;
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
 *
 * Observation kind is set by `opts.asType` (default `'span'` for back-compat with existing callers).
 * With `'generation'`, Langfuse recognises token/cost fields and renders them in its cost/usage
 * views, and the `usage` callback is invoked on success to populate `usageDetails`, `costDetails`,
 * and `model`. Those fields are generation-only and are NOT applied to spans.
 */
export async function traceSessionTurn<T>(
  fn: () => Promise<T>,
  opts: TraceSessionTurnOptions<T>,
): Promise<T> {
  const { name, sessionId, input, metadata, tags, usage, asType = 'span' } = opts;

  // Self-root: cleared OTel context + trace-level attributes around the observation factory.
  // Generic so each branch keeps its concrete observation type (LangfuseGeneration / LangfuseSpan).
  const startSelfRooted = <O>(create: () => O): O =>
    context.with(ROOT_CONTEXT, () =>
      propagateAttributes(
        {
          traceName: name,
          sessionId,
          tags,
          metadata: stringifyMetadata(metadata),
        },
        create,
      ),
    );

  const errorAttrs = (e: unknown) => ({
    level: 'ERROR' as const,
    statusMessage: e instanceof Error ? (e.stack ?? e.message) : String(e),
  });

  if (asType === 'generation') {
    const generation = startSelfRooted(() =>
      startObservation(
        name,
        { input, metadata } as LangfuseGenerationAttributes,
        { asType: 'generation' },
      ),
    );
    try {
      const result = await fn();
      const u = usage?.(result);
      generation.update({
        output: result,
        ...(u?.usageDetails ? { usageDetails: u.usageDetails } : {}),
        ...(u?.costDetails ? { costDetails: u.costDetails } : {}),
        ...(u?.model ? { model: u.model } : {}),
      } as LangfuseGenerationAttributes);
      return result;
    } catch (e) {
      generation.update(errorAttrs(e) as LangfuseGenerationAttributes);
      throw e;
    } finally {
      generation.end();
    }
  }

  // Default 'span' — original behavior for existing callers. `usage` is generation-only and is
  // intentionally NOT applied here, so a span never receives token/cost fields.
  const span = startSelfRooted(() =>
    startObservation(
      name,
      { input, metadata } as LangfuseSpanAttributes,
      { asType: 'span' },
    ),
  );
  try {
    const result = await fn();
    span.update({ output: result } as LangfuseSpanAttributes);
    return result;
  } catch (e) {
    span.update(errorAttrs(e) as LangfuseSpanAttributes);
    throw e;
  } finally {
    span.end();
  }
}
