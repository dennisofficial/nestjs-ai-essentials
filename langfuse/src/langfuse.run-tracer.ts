import {
  LangfuseAgent,
  LangfuseChain,
  LangfuseEmbedding,
  LangfuseEvaluator,
  LangfuseEventAttributes,
  LangfuseGeneration,
  LangfuseGenerationAttributes,
  LangfuseGuardrail,
  LangfuseObservation,
  LangfuseObservationType,
  LangfuseRetriever,
  LangfuseSpan,
  LangfuseSpanAttributes,
  LangfuseTool,
  propagateAttributes,
  startObservation,
} from '@langfuse/tracing';
import { LangfuseCallbackHandler, LangfuseCallbackOptions } from './langfuse.callback';
import { cleanStackTrace, parseParentContext } from './langfuse.utils';
import { serializeInputsOutputs } from './langfuse.serializer';
import { handleMedia } from './langfuse.media';

interface TypeToObserver {
  span: LangfuseSpan;
  generation: LangfuseGeneration;
  embedding: LangfuseEmbedding;
  agent: LangfuseAgent;
  tool: LangfuseTool;
  chain: LangfuseChain;
  retriever: LangfuseRetriever;
  evaluator: LangfuseEvaluator;
  guardrail: LangfuseGuardrail;
}

interface TypeToAttributes {
  span: LangfuseSpanAttributes;
  generation: LangfuseGenerationAttributes;
  embedding: LangfuseGenerationAttributes;
  agent: LangfuseSpanAttributes;
  tool: LangfuseSpanAttributes;
  chain: LangfuseSpanAttributes;
  retriever: LangfuseSpanAttributes;
  evaluator: LangfuseSpanAttributes;
  guardrail: LangfuseSpanAttributes;
}

type RootTraceAttributes = {
  userId?: string;
  sessionId?: string;
  tags?: string[];
  version?: string;
  traceMetadata?: Record<string, unknown>;
};

type TraceCaptureOptions = {
  captureInput?: boolean;
  captureOutput?: boolean;
};

const stringifyMetadata = (
  metadata?: Record<string, unknown>,
): Record<string, string> | undefined => {
  if (!metadata) return undefined;

  const entries = Object.entries(metadata);
  if (entries.length === 0) return undefined;

  return Object.fromEntries(
    entries.map(([key, value]) => [key, typeof value === 'string' ? value : JSON.stringify(value)]),
  );
};

export class RunTracer {
  static from = (options?: LangfuseCallbackOptions): LangfuseCallbackHandler => {
    return new LangfuseCallbackHandler(options);
  };

  static traceAsync<OType extends Exclude<LangfuseObservationType, 'event'>>(
    params: {
      name: string;
    } & RootTraceAttributes &
      Omit<LangfuseSpanAttributes, 'output'>,
  ): TypeToObserver[OType] {
    const { name, userId, sessionId, tags, version, traceMetadata, ...attributes } = params;

    const create = () =>
      startObservation(name, attributes, {
        asType: 'span',
        parentSpanContext: parseParentContext(undefined),
      });

    return propagateAttributes(
      {
        traceName: name,
        userId,
        sessionId,
        tags,
        version,
        metadata: stringifyMetadata(traceMetadata),
      },
      create,
    );
  }

  static async traceEvent<T>(
    output: T,
    params: {
      name: string;
      parent?: LangfuseObservation;
    } & RootTraceAttributes &
      Omit<LangfuseEventAttributes, 'output'>,
  ): Promise<T> {
    const { name, parent, userId, sessionId, tags, version, traceMetadata, ...attributes } = params;

    const serializedInput = await serializeInputsOutputs(params.input);
    const serializedOutput = await serializeInputsOutputs(output);

    const create = async () =>
      startObservation(
        name,
        {
          ...attributes,
          input: serializedInput,
          output: serializedOutput,
        },
        { asType: 'event', parentSpanContext: parseParentContext(parent) },
      );

    const observer = parent
      ? await create()
      : await propagateAttributes(
          {
            traceName: name,
            userId,
            sessionId,
            tags,
            version,
            metadata: stringifyMetadata(traceMetadata),
          },
          create,
        );
    observer.end();
    return output;
  }

  static async trace<T, OType extends Exclude<LangfuseObservationType, 'event'>>(
    func: (run_tree: TypeToObserver[OType]) => T | Promise<T>,
    params: {
      name: string;
      parent?: LangfuseObservation;
      type?: OType;
    } & RootTraceAttributes &
      TraceCaptureOptions &
      Omit<TypeToAttributes[OType], 'output'>,
  ): Promise<T> {
    const {
      name,
      parent,
      type,
      userId,
      sessionId,
      tags,
      version,
      traceMetadata,
      captureInput = true,
      captureOutput = true,
      ...attributes
    } = params;

    const {
      input: initialInput,
      metadata: initialMetadata,
      output: _initialOutput,
      ...startAttributes
    } = attributes as TypeToAttributes[OType] & {
      input?: unknown;
      output?: unknown;
      metadata?: Record<string, unknown>;
    };

    const create = () =>
      startObservation(name, startAttributes, {
        // @ts-ignore This is correct. The package has unique typings
        asType: type ?? 'agent',
        parentSpanContext: parseParentContext(parent),
      }) as TypeToObserver[OType];

    const observer: TypeToObserver[OType] = parent
      ? create()
      : propagateAttributes(
          {
            traceName: name,
            userId,
            sessionId,
            tags,
            version,
            metadata: stringifyMetadata(traceMetadata),
          },
          create,
        );

    try {
      if (captureInput) {
        // Handle media in input
        try {
          const transformedInput = await handleMedia(
            initialInput,
            'input',
            observer.traceId,
            observer.id,
          );
          const input = await serializeInputsOutputs(transformedInput);
          observer.update({ input });
        } catch (e) {
          const input = await serializeInputsOutputs(initialInput);
          observer.update({ input });
          console.error('Error handling media in input', e);
        }
      }

      // Execute a wrapped function and await it if it is async
      const result = await new Promise<T>((resolve, reject) => {
        const result = func(observer);
        if (result instanceof Promise) result.then(resolve).catch(reject);
        else resolve(result);
      });

      if (captureOutput) {
        try {
          const transformedOutput = await handleMedia(
            result,
            'output',
            observer.traceId,
            observer.id,
          );
          const output = await serializeInputsOutputs(transformedOutput);
          observer.update({ output });
        } catch (e) {
          const output = await serializeInputsOutputs(result);
          observer.update({ output });
          console.error('Error handling media in output', e);
        }
      }

      try {
        const transformedMetadata: typeof initialMetadata = (await handleMedia(
          initialMetadata,
          'metadata',
          observer.traceId,
          observer.id,
        )) as any;
        observer.update({ metadata: transformedMetadata });
      } catch (e) {
        console.error('Error handling media in metadata', e);
      }

      // Return wrapped function result
      return result;
    } catch (e: any) {
      if (e instanceof Error) {
        observer.update({
          statusMessage: cleanStackTrace(e?.stack) ?? e?.message ?? 'Unknown Error',
          level: 'ERROR',
        });
      } else {
        observer.update({
          statusMessage: 'Unknown Error',
          level: 'ERROR',
        });
      }

      // Rethrow error
      throw e;
    } finally {
      // Create Stack Trace line for langfuse. This is to find the run in the codebase if needed
      const traceLine = String(new Error().stack?.split('\n').slice(3)[0]);
      const cleanTrace = traceLine.replaceAll(process.cwd(), '').replaceAll('at ', '').trim();

      try {
        const transformedMetadata = await handleMedia(
          initialMetadata,
          'metadata',
          observer.traceId,
          observer.id,
        );
        observer.update({ metadata: await serializeInputsOutputs(transformedMetadata) });
      } catch (e) {
        console.error('Error handling media in metadata', e);
      }

      // Add trace to langfuse
      observer.update({ metadata: { trace: cleanTrace } });
      observer.end();
    }
  }
}
