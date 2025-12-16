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

export class RunTracer {
  static from = (options?: LangfuseCallbackOptions): LangfuseCallbackHandler => {
    return new LangfuseCallbackHandler(options);
  };

  static traceAsync<OType extends Exclude<LangfuseObservationType, 'event'>>(
    params: {
      name: string;
    } & Omit<LangfuseSpanAttributes, 'output'>,
  ): TypeToObserver[OType] {
    const { name, ...attributes } = params;

    const observer = startObservation(name, attributes, {
      asType: 'span',
      parentSpanContext: parseParentContext(undefined),
    });
    observer.updateTrace({ name });
    return observer;
  }

  static async traceEvent<T>(
    output: T,
    params: {
      name: string;
      parent?: LangfuseObservation;
    } & Omit<LangfuseEventAttributes, 'output'>,
  ): Promise<T> {
    const { name, parent, ...attributes } = params;

    const parentSpanContext = parseParentContext(parent);
    const observer = startObservation(
      name,
      {
        ...attributes,
        output: await serializeInputsOutputs(output),
        input: await serializeInputsOutputs(params.input),
      },
      { asType: 'event', parentSpanContext },
    );
    if (!parent) {
      observer.updateTrace({ name });
    }
    observer.end();
    return output;
  }

  static async trace<T, OType extends Exclude<LangfuseObservationType, 'event'>>(
    func: (run_tree: TypeToObserver[OType]) => T | Promise<T>,
    params: {
      name: string;
      parent?: LangfuseObservation;
      type?: OType;
    } & Omit<TypeToAttributes[OType], 'output'>,
  ): Promise<T> {
    const { name, parent, type, ...attributes } = params;

    const observer: TypeToObserver[OType] = startObservation(name, attributes, {
      // @ts-ignore This is correct. The package has unique typings
      asType: type ?? 'agent',
      parentSpanContext: parseParentContext(parent),
    });
    if (!parent) observer.updateTrace({ name });

    try {
      // Handle media in input
      try {
        const transformedInput = await handleMedia(
          params.input,
          'input',
          observer.traceId,
          observer.id,
        );
        const input = await serializeInputsOutputs(transformedInput);
        if (!parent) observer.updateTrace({ input });
        observer.update({ input: await serializeInputsOutputs(transformedInput) });
      } catch (e) {
        const input = await serializeInputsOutputs(params.input);
        if (!parent) observer.updateTrace({ input });
        observer.update({ input: await serializeInputsOutputs(params.input) });
        console.error('Error handling media in input', e);
      }

      // Execute a wrapped function and await it if it is async
      const result = await new Promise<T>((resolve, reject) => {
        const result = func(observer);
        if (result instanceof Promise) result.then(resolve).catch(reject);
        else resolve(result);
      });

      try {
        const transformedOutput = await handleMedia(
          result,
          'output',
          observer.traceId,
          observer.id,
        );
        const output = await serializeInputsOutputs(transformedOutput);
        observer.update({ output });
        if (!parent) observer.updateTrace({ output });
      } catch (e) {
        const output = await serializeInputsOutputs(result);
        observer.update({ output });
        if (!parent) observer.updateTrace({ output });
        console.error('Error handling media in output', e);
      }

      try {
        const transformedMetadata: typeof params.metadata = (await handleMedia(
          params.metadata,
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
          params.metadata,
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
