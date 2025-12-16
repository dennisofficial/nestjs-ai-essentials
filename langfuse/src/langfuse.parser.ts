import { isDefined } from 'class-validator';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  FunctionMessage,
  FunctionMessageChunk,
  HumanMessage,
  HumanMessageChunk,
  SystemMessage,
  SystemMessageChunk,
  ToolMessage,
  ToolMessageChunk,
} from '@langchain/core/messages';
import { LangfuseMedia } from '@langfuse/core';
import { ChatPromptValue } from '@langchain/core/prompt_values';
import { LlmMessage } from './langfuse.callback';

export const circularTransformer = async (
  data: unknown,
  transformers: Array<(data: any, debug?: boolean) => any>,
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

export const serializeBaseMessage = (message: any, debug?: boolean): LlmMessage | undefined => {
  if (!BaseMessage.isInstance(message)) return;

  const response: LlmMessage = { role: message.type };

  if (isDefined(message.content) && message.content.length) {
    response.content = message.content;
  }

  if (isSystemMessage(message)) {
    response.role = 'system';
  }

  if (isHumanMessage(message)) {
    response.role = 'user';
  }

  if (isFunctionMessage(message)) {
    response.role = message.name ?? response.role;
  }

  if (isAiMessage(message)) {
    response.role = 'assistant';

    if (message.tool_calls?.length) {
      if (isDefined(message.content) && message.content.length) {
        response.content = undefined;
      }

      response.additional_kwargs = { tool_calls: message.tool_calls };
    }
  }

  if (isToolMessage(message)) {
    response.role = message.name ?? response.role;

    response.additional_kwargs = {
      tool_call_id: message.tool_call_id,
      tool_call_name: message.name,
    };
  }

  return response;
};

const isAiMessage = (obj: any): obj is AIMessage | AIMessageChunk =>
  AIMessage.isInstance(obj) || AIMessageChunk.isInstance(obj);

const isToolMessage = (obj: any): obj is ToolMessage | ToolMessageChunk =>
  ToolMessage.isInstance(obj) || ToolMessageChunk.isInstance(obj);

const isFunctionMessage = (obj: any): obj is FunctionMessage | FunctionMessageChunk =>
  FunctionMessage.isInstance(obj) || FunctionMessageChunk.isInstance(obj);

const isHumanMessage = (obj: any): obj is HumanMessage | HumanMessageChunk =>
  HumanMessage.isInstance(obj) || HumanMessageChunk.isInstance(obj);

const isSystemMessage = (obj: any): obj is SystemMessage | SystemMessageChunk =>
  SystemMessage.isInstance(obj) || SystemMessageChunk.isInstance(obj);
