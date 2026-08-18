import { isDefined } from 'class-validator';
import { LangfuseMedia } from '@langfuse/core';
import { ChatPromptValue } from '@langchain/core/prompt_values';

export type CircularTransformerFn = (field: any, isDebug?: boolean) => any | Promise<any>;

export const circularTransformer = async (
  data: unknown,
  transformers: Array<CircularTransformerFn>,
  debug = false,
): Promise<unknown> => {
  const visited = new Map();

  const debugLog = (...any: any[]) => {
    if (debug) console.debug(...any);
  };

  const transform = async (data: unknown): Promise<unknown> => {
    debugLog(data);

    if (!isDefined(data)) return data;
    if (visited.has(data)) return undefined;

    // Transform each array item
    if (Array.isArray(data)) {
      visited.set(data, data);
      const transformed = await Promise.all(data.map((item) => transform(item)));
      visited.set(data, transformed);
      return transformed;
    }

    // Try to find an appropriate transformer
    for (const transformer of transformers) {
      let result = transformer(data, debug);

      if (result instanceof Promise) {
        result = await result;
      }

      if (isDefined(result)) {
        if (typeof data === 'object') {
          visited.set(data, result);
        }
        debugLog('Transformed:', result);
        return result;
      }
    }

    // Transform object keys
    if (typeof data === 'object' && isDefined(data)) {
      visited.set(data, data);
      const entriesPromises = Object.entries(data).map(async ([k, v]) => [k, await transform(v)]);
      const entries = await Promise.all(entriesPromises);
      const result = Object.fromEntries(entries);
      visited.set(data, result);
      return result;
    }

    // Other non-object types
    return data;
  };

  return await transform(data);
};

export const serializePrompts = async (chatValue: any): Promise<any> => {
  if (!(chatValue instanceof ChatPromptValue)) return;

  return chatValue.messages.map((message) => [message.type, message.content]);
};

export const serializeLangfuseMedia = async (media: any): Promise<string | null> => {
  if (media instanceof LangfuseMedia) {
    const mediaTag = await media.getTag();
    if (!mediaTag) return 'LangfuseMedia';
    return media.getTag();
  }
  return null;
};
