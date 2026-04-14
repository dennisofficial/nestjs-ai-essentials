// https://github.com/langfuse/langfuse-js/blob/main/packages/langchain/src/CallbackHandler.ts

import type { AgentAction, AgentFinish } from '@langchain/core/agents';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { Document } from '@langchain/core/documents';
import type { Serialized } from '@langchain/core/load/serializable';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  type BaseMessageFields,
  type MessageContent,
  type UsageMetadata,
} from '@langchain/core/messages';
import type { Generation, LLMResult } from '@langchain/core/outputs';
import type { ChainValues } from '@langchain/core/utils/types';
import { getGlobalLogger } from '@langfuse/core';
import {
  type LangfuseGenerationAttributes,
  type LangfuseObservation,
  type LangfuseSpanAttributes,
  propagateAttributes,
  startActiveObservation,
} from '@langfuse/tracing';

import { cleanStackTrace } from './langfuse.utils';

const LANGSMITH_HIDDEN_TAG = 'langsmith:hidden';

type LangfusePrompt = {
  name: string;
  version: number;
  isFallback: boolean;
};

type ObservationType =
  | 'agent'
  | 'chain'
  | 'generation'
  | 'guardrail'
  | 'retriever'
  | 'span'
  | 'tool';

export interface LangfuseCallbackOptions {
  parent?: LangfuseObservation;
  userId?: string;
  sessionId?: string;
  tags?: string[];
  version?: string;
  traceMetadata?: Record<string, unknown>;
}

export type LlmMessage = {
  role?: string;
  content?: BaseMessageFields['content'];
  additional_kwargs?: Record<string, any>;
};

export type AnonymousLlmMessage = {
  content: BaseMessageFields['content'];
  additional_kwargs?: Record<string, any>;
};

class LangfuseCallbackHandler extends BaseCallbackHandler {
  name = 'LangfuseCallbackHandler';

  public root?: LangfuseObservation;
  public last_trace_id: string | null = null;

  private readonly userId?: string;
  private readonly version?: string;
  private readonly sessionId?: string;
  private readonly tags: string[];
  private readonly traceMetadata?: Record<string, unknown>;

  private completionStartTimes: Record<string, Date> = {};
  private promptToParentRunMap = new Map<string, LangfusePrompt>();
  private runMap: Map<string, LangfuseObservation> = new Map();
  private skippedRunMap: Map<string, string> = new Map();

  constructor(params?: LangfuseCallbackOptions) {
    super({});

    this.root = params?.parent;
    this.sessionId = params?.sessionId;
    this.userId = params?.userId;
    this.tags = params?.tags ?? [];
    this.traceMetadata = params?.traceMetadata;
    this.version = params?.version;
  }

  get logger() {
    return getGlobalLogger();
  }

  async handleLLMNewToken(
    _token: string,
    _idx: unknown,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _fields?: unknown,
  ): Promise<void> {
    if (runId && !(runId in this.completionStartTimes)) {
      this.logger.debug(`LLM first streaming token: ${runId}`);
      this.completionStartTimes[runId] = new Date();
    }
  }

  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runType?: string,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, chain, tags, 'Langchain Run');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Chain start with Id: ${runId}`);
      this.registerLangfusePrompt(parentRunId, metadata);

      let finalInput: string | ChainValues = inputs;
      if (
        typeof inputs === 'object' &&
        'input' in inputs &&
        Array.isArray(inputs['input']) &&
        inputs['input'].every((m: unknown) => m instanceof BaseMessage)
      ) {
        finalInput = inputs['input'].map((m: BaseMessage) => this.extractChatMessageContent(m));
      } else if (
        typeof inputs === 'object' &&
        'messages' in inputs &&
        Array.isArray(inputs['messages']) &&
        inputs['messages'].every((m: unknown) => m instanceof BaseMessage)
      ) {
        finalInput = inputs['messages'].map((m: BaseMessage) => this.extractChatMessageContent(m));
      } else if (
        typeof inputs === 'object' &&
        'content' in inputs &&
        typeof inputs['content'] === 'string' &&
        Object.keys(inputs).length === 1
      ) {
        finalInput = inputs['content'];
      }

      this.startRootAwareObservation({
        type: this.getChainObservationType(runType),
        runName,
        runId,
        parentRunId,
        tags,
        metadata,
        attributes: {
          input: finalInput,
          level: runType === 'parser' ? 'DEBUG' : undefined,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleAgentAction(action: AgentAction, runId: string, parentRunId?: string): Promise<void> {
    try {
      const runName = action.tool;
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Agent action ${action.tool} with ID: ${runId}`);
      this.startRootAwareObservation({
        type: 'tool',
        runId,
        parentRunId,
        runName,
        attributes: {
          input: action,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleAgentEnd(action: AgentFinish, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Agent finish with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: { output: action },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleChainError(err: unknown, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Chain error: ${String(err)} with ID: ${runId}`);

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage:
            cleanStackTrace(err instanceof Error ? err.stack : undefined) ?? String(err),
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleGenerationStart(
    llm: Serialized,
    messages: (LlmMessage | MessageContent | AnonymousLlmMessage)[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    const runName = this.getName(name, llm, tags, 'Langchain Generation');
    console.log(`handleGenerationStart: ${runName}`, this.shouldTrace(runName, runId, parentRunId));

    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    this.logger.debug(`Generation start with ID: ${runId} and parentRunId ${parentRunId}`);

    const modelParameters: Record<string, string | number> = {};
    const invocationParams = extraParams?.['invocation_params'];

    for (const [key, value] of Object.entries({
      temperature: (invocationParams as any)?.temperature,
      max_tokens: (invocationParams as any)?.max_tokens,
      top_p: (invocationParams as any)?.top_p,
      frequency_penalty: (invocationParams as any)?.frequency_penalty,
      presence_penalty: (invocationParams as any)?.presence_penalty,
      request_timeout: (invocationParams as any)?.request_timeout,
    })) {
      if (value !== undefined && value !== null) {
        modelParameters[key] = value;
      }
    }

    interface InvocationParams {
      model?: string;
    }

    const invocationParamsModelName = (
      extraParams?.invocation_params as InvocationParams | undefined
    )?.model;
    const metadataModelName =
      metadata && 'ls_model_name' in metadata ? (metadata['ls_model_name'] as string) : undefined;
    const extractedModelName = invocationParamsModelName ?? metadataModelName;

    const registeredPrompt = this.promptToParentRunMap.get(
      this.resolveParentRunId(parentRunId) ?? 'root',
    );
    if (registeredPrompt && parentRunId) {
      this.deregisterLangfusePrompt(parentRunId);
    }

    this.startRootAwareObservation({
      type: 'generation',
      runId,
      parentRunId,
      metadata,
      tags,
      runName,
      attributes: {
        input: messages,
        model: extractedModelName,
        modelParameters,
        prompt: registeredPrompt,
      },
    });
  }

  async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      this.logger.debug(`Chat model start with ID: ${runId}`);

      const prompts = messages.flatMap((message) =>
        message.map((m) => this.extractChatMessageContent(m)),
      );

      await this.handleGenerationStart(
        llm,
        prompts,
        runId,
        parentRunId,
        extraParams,
        tags,
        metadata,
        name,
      );
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleChainEnd(outputs: ChainValues, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Chain end with ID: ${runId}`);

      let finalOutput: ChainValues | string = outputs;
      if (
        typeof outputs === 'object' &&
        'output' in outputs &&
        typeof outputs['output'] === 'string'
      ) {
        finalOutput = outputs['output'];
      } else if (
        typeof outputs === 'object' &&
        'messages' in outputs &&
        Array.isArray(outputs['messages']) &&
        outputs['messages'].every((m: unknown) => m instanceof BaseMessage)
      ) {
        finalOutput = {
          messages: outputs.messages.map((message: BaseMessage) =>
            this.extractChatMessageContent(message),
          ),
        };
      }

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          output: finalOutput,
        },
      });
      this.deregisterLangfusePrompt(runId);
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      this.logger.debug(`LLM start with ID: ${runId}`);
      await this.handleGenerationStart(
        llm,
        prompts,
        runId,
        parentRunId,
        extraParams,
        tags,
        metadata,
        name,
      );
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleToolStart(
    tool: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, tool, tags, 'Tool execution');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Tool start with ID: ${runId}`);
      this.startRootAwareObservation({
        type: 'tool',
        runId,
        parentRunId,
        runName,
        attributes: {
          input,
        },
        metadata,
        tags,
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, retriever, tags, 'Retriever');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Retriever start with ID: ${runId}`);
      this.startRootAwareObservation({
        type: 'retriever',
        runId,
        parentRunId,
        runName,
        attributes: {
          input: query,
        },
        tags,
        metadata,
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleRetrieverEnd(
    documents: Document<Record<string, any>>[],
    runId: string,
    _parentRunId?: string,
  ): Promise<void> {
    try {
      this.logger.debug(`Retriever end with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: {
          output: documents,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleRetrieverError(err: unknown, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Retriever error: ${String(err)} with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage:
            cleanStackTrace(err instanceof Error ? err.stack : undefined) ?? String(err),
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleToolEnd(output: string, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Tool end with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: { output },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleToolError(err: unknown, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Tool error ${String(err)} with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage:
            cleanStackTrace(err instanceof Error ? err.stack : undefined) ?? String(err),
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleLLMEnd(output: LLMResult, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`LLM end with ID: ${runId}`);
      console.log(
        '[langfuse llm end:start]',
        JSON.stringify({
          runId,
          hasRunMap: this.runMap.has(runId),
          generationBatches: Array.isArray(output.generations) ? output.generations.length : null,
          lastBatchLength:
            Array.isArray(output.generations) && output.generations.length > 0
              ? (output.generations[output.generations.length - 1]?.length ?? null)
              : null,
          hasLlmOutput: Boolean(output.llmOutput),
        }),
      );

      const lastResponse =
        output.generations[output.generations.length - 1][
          output.generations[output.generations.length - 1].length - 1
        ];
      const llmUsage =
        this.extractUsageMetadata(lastResponse) ?? output.llmOutput?.['tokenUsage'] ?? {};
      const modelName = this.extractModelNameFromMetadata(lastResponse);

      const usageDetails: Record<string, unknown> = {
        input:
          llmUsage?.input_tokens ??
          ('promptTokens' in llmUsage ? llmUsage.promptTokens : undefined),
        output:
          llmUsage?.output_tokens ??
          ('completionTokens' in llmUsage ? llmUsage.completionTokens : undefined),
        total:
          llmUsage?.total_tokens ?? ('totalTokens' in llmUsage ? llmUsage.totalTokens : undefined),
      };

      if (llmUsage && 'input_token_details' in llmUsage) {
        for (const [key, val] of Object.entries(llmUsage.input_token_details ?? {})) {
          usageDetails[`input_${key}`] = val;

          if (
            'input' in usageDetails &&
            typeof val === 'number' &&
            typeof usageDetails.input === 'number'
          ) {
            usageDetails.input = Math.max(0, usageDetails.input - val);
          }
        }
      }

      if (llmUsage && 'output_token_details' in llmUsage) {
        for (const [key, val] of Object.entries(llmUsage.output_token_details ?? {})) {
          usageDetails[`output_${key}`] = val;

          if (
            'output' in usageDetails &&
            typeof val === 'number' &&
            typeof usageDetails.output === 'number'
          ) {
            usageDetails.output = Math.max(0, usageDetails.output - val);
          }
        }
      }

      const extractedOutput =
        'message' in lastResponse
          ? this.extractChatMessageContent(lastResponse.message as BaseMessage)
          : lastResponse.text;

      console.log(
        '[langfuse llm end:before-end]',
        JSON.stringify({
          runId,
          hasRunMap: this.runMap.has(runId),
          modelName,
          outputType: typeof extractedOutput,
          hasCompletionStartTime: runId in this.completionStartTimes,
        }),
      );

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          model: modelName,
          output: extractedOutput,
          completionStartTime:
            runId in this.completionStartTimes ? this.completionStartTimes[runId] : undefined,
          usageDetails,
        },
      });

      if (runId in this.completionStartTimes) {
        delete this.completionStartTimes[runId];
      }
    } catch (e) {
      console.error(
        '[langfuse llm end:error]',
        JSON.stringify({
          runId,
          hasRunMap: this.runMap.has(runId),
          message: e instanceof Error ? e.message : String(e),
        }),
      );
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleLLMError(err: unknown, runId: string, _parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`LLM error ${String(err)} with ID: ${runId}`);
      console.error(
        '[langfuse llm error]',
        JSON.stringify({
          runId,
          hasRunMap: this.runMap.has(runId),
          message: err instanceof Error ? err.message : String(err),
        }),
      );
      const azureRefusalError = this.parseAzureRefusalError(err);

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage:
            (cleanStackTrace(err instanceof Error ? err.stack : undefined) ?? String(err)) +
            azureRefusalError,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  private registerLangfusePrompt(parentRunId?: string, metadata?: Record<string, unknown>): void {
    if (metadata && 'langfusePrompt' in metadata && parentRunId) {
      this.promptToParentRunMap.set(parentRunId, metadata.langfusePrompt as LangfusePrompt);
    }
  }

  private deregisterLangfusePrompt(runId: string): void {
    this.promptToParentRunMap.delete(runId);
  }

  private startRootAwareObservation(params: {
    type?: ObservationType;
    runName: string;
    runId: string;
    parentRunId?: string;
    attributes: LangfuseGenerationAttributes;
    metadata?: Record<string, unknown>;
    tags?: string[];
  }): LangfuseObservation {
    const { runName, parentRunId, tags, metadata } = params;
    const resolvedParentRunId = this.resolveParentRunId(parentRunId);
    const shouldPropagateTraceAttributes = !resolvedParentRunId && !this.root;

    const startObservation = () =>
      this.startAndRegisterOtelSpan({
        ...params,
        parentRunId: resolvedParentRunId,
      });

    if (!shouldPropagateTraceAttributes) {
      return startObservation();
    }

    const traceTags = [...new Set([...(tags ?? []), ...this.tags])];
    const traceUserId =
      metadata && 'langfuseUserId' in metadata && typeof metadata.langfuseUserId === 'string'
        ? metadata.langfuseUserId
        : this.userId;
    const traceSessionId =
      metadata && 'langfuseSessionId' in metadata && typeof metadata.langfuseSessionId === 'string'
        ? metadata.langfuseSessionId
        : this.sessionId;
    const traceMetadata = this.stringifyMetadata(
      this.joinTagsAndMetaData(undefined, this.traceMetadata, metadata),
    );

    let observation!: LangfuseObservation;
    propagateAttributes(
      {
        traceName: runName,
        tags: traceTags.length > 0 ? traceTags : undefined,
        userId: traceUserId,
        sessionId: traceSessionId,
        metadata: traceMetadata,
        version: this.version,
      },
      () => {
        observation = startObservation();
      },
    );

    return observation;
  }

  private startAndRegisterOtelSpan(params: {
    type?: ObservationType;
    runName: string;
    runId: string;
    parentRunId?: string;
    attributes: LangfuseGenerationAttributes;
    metadata?: Record<string, unknown>;
    tags?: string[];
  }): LangfuseObservation {
    const { type = 'span', runName, runId, parentRunId, attributes, metadata, tags } = params;

    const observationAttributes = {
      version: this.version,
      metadata: this.joinTagsAndMetaData(tags, metadata),
      level: tags?.includes(LANGSMITH_HIDDEN_TAG) ? 'DEBUG' : attributes.level,
      ...attributes,
    } as LangfuseGenerationAttributes;

    const observation =
      type === 'span'
        ? (startActiveObservation(
            runName,
            (activeObservation) => {
              activeObservation.update(observationAttributes as LangfuseSpanAttributes);
              return activeObservation;
            },
            {
              parentSpanContext: this.getParentSpanContext(parentRunId),
              endOnExit: false,
            },
          ) as LangfuseObservation)
        : (startActiveObservation(
            runName,
            (activeObservation) => {
              activeObservation.update(observationAttributes as LangfuseSpanAttributes);
              return activeObservation;
            },
            {
              asType: type,
              parentSpanContext: this.getParentSpanContext(parentRunId),
              endOnExit: false,
            } as any,
          ) as LangfuseObservation);

    if (!parentRunId && !this.root) {
      this.root = observation;
    }

    this.runMap.set(runId, observation);
    return observation;
  }

  private handleOtelSpanEnd(params: {
    runId: string;
    attributes?: LangfuseGenerationAttributes | LangfuseSpanAttributes;
  }): void {
    const { runId, attributes = {} } = params;

    const span = this.runMap.get(runId);
    if (!span) {
      console.warn(
        '[langfuse otel end:missing-span]',
        JSON.stringify({
          runId,
          remainingRunMapSize: this.runMap.size,
          attributeKeys: Object.keys(attributes),
        }),
      );
      return;
    }

    console.log(
      '[langfuse otel end]',
      JSON.stringify({
        runId,
        traceId: span.traceId,
        attributeKeys: Object.keys(attributes),
      }),
    );

    if ('update' in span) {
      span.update(attributes as LangfuseSpanAttributes);
    }
    span.end();
    this.last_trace_id = span.traceId;
    this.runMap.delete(runId);
    this.skippedRunMap.delete(runId);
  }

  private parseAzureRefusalError(err: unknown): string {
    let azureRefusalError = '';
    if (err && typeof err === 'object' && 'error' in err) {
      try {
        azureRefusalError = '\n\nError details:\n' + JSON.stringify((err as any).error, null, 2);
      } catch {}
    }
    return azureRefusalError;
  }

  private joinTagsAndMetaData(
    tags?: string[],
    metadata1?: Record<string, unknown>,
    metadata2?: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    const finalDict: Record<string, unknown> = {};
    if (tags && tags.length > 0) {
      finalDict.tags = tags;
    }
    if (metadata1) {
      Object.assign(finalDict, metadata1);
    }
    if (metadata2) {
      Object.assign(finalDict, metadata2);
    }
    return this.stripLangfuseKeysFromMetadata(finalDict);
  }

  private stripLangfuseKeysFromMetadata(
    metadata?: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!metadata) return;

    const langfuseKeys = ['langfusePrompt', 'langfuseSessionId', 'langfuseUserId'];
    return Object.fromEntries(
      Object.entries(metadata).filter(([key]) => !langfuseKeys.includes(key)),
    );
  }

  private stringifyMetadata(
    metadata?: Record<string, unknown>,
  ): Record<string, string> | undefined {
    if (!metadata) return;

    const entries = Object.entries(metadata);
    if (entries.length === 0) return;

    return Object.fromEntries(
      entries.map(([key, value]) => [
        key,
        typeof value === 'string' ? value : JSON.stringify(value),
      ]),
    );
  }

  private extractUsageMetadata(generation: Generation): UsageMetadata | undefined {
    try {
      return 'message' in generation &&
        (generation.message instanceof AIMessage || generation.message instanceof AIMessageChunk)
        ? generation.message.usage_metadata
        : undefined;
    } catch (err) {
      this.logger.debug(`Error extracting usage metadata: ${err}`);
      return;
    }
  }

  private extractModelNameFromMetadata(generation: any): string | undefined {
    try {
      return 'message' in generation &&
        (generation.message instanceof AIMessage || generation.message instanceof AIMessageChunk)
        ? generation.message.response_metadata.model_name
        : undefined;
    } catch {}
  }

  private extractChatMessageContent(
    message: BaseMessage,
  ): LlmMessage | AnonymousLlmMessage | MessageContent {
    let response: LlmMessage | AnonymousLlmMessage | Record<string, unknown> | undefined;

    if (message.type === 'human') {
      response = { content: message.content, role: 'user' };
    } else if (message.type === 'generic') {
      response = { content: message.content, role: 'human' };
    } else if (message.type === 'ai') {
      response = { content: message.content, role: 'assistant' };

      if (
        'tool_calls' in message &&
        Array.isArray(message.tool_calls) &&
        (message.tool_calls?.length ?? 0) > 0
      ) {
        (response as any).tool_calls = message.tool_calls;
      }
      if ('additional_kwargs' in message && 'tool_calls' in message.additional_kwargs) {
        (response as any).tool_calls = message.additional_kwargs.tool_calls;
      }
    } else if (message.type === 'system') {
      response = { content: message.content, role: 'system' };
    } else if (message.type === 'function' || message.type === 'tool') {
      response = {
        content: message.content,
        additional_kwargs: message.additional_kwargs,
        role: message.name,
      };
    } else if (!message.name) {
      response = { content: message.content };
    } else {
      response = {
        role: message.name,
        content: message.content,
      };
    }

    if (
      (message.additional_kwargs.function_call || message.additional_kwargs.tool_calls) &&
      (response as any).tool_calls === undefined
    ) {
      return { ...response, additional_kwargs: message.additional_kwargs };
    }

    return response;
  }

  private getChainObservationType(runType?: string): ObservationType {
    if (runType === 'agent') return 'agent';
    if (runType === 'parser') return 'guardrail';
    if (runType === 'prompt') return 'span';
    return 'chain';
  }

  private getParentSpanContext(parentRunId?: string) {
    const resolvedParentRunId = this.resolveParentRunId(parentRunId);
    if (resolvedParentRunId) {
      return this.runMap.get(resolvedParentRunId)?.otelSpan.spanContext();
    }
    return this.root?.otelSpan.spanContext();
  }

  private resolveParentRunId(parentRunId?: string): string | undefined {
    if (!parentRunId) return;

    let resolvedRunId: string | undefined = parentRunId;
    const visited = new Set<string>();

    while (resolvedRunId && this.skippedRunMap.has(resolvedRunId) && !visited.has(resolvedRunId)) {
      visited.add(resolvedRunId);
      resolvedRunId = this.skippedRunMap.get(resolvedRunId);
    }

    return resolvedRunId;
  }

  private shouldTrace(runName: string, runId: string, parentRunId?: string): boolean {
    if (!parentRunId) return true;
    if (runName.startsWith('Runnable')) {
      this.skippedRunMap.set(runId, parentRunId);
      return false;
    }
    return true;
  }

  private getName(
    runName: string | undefined,
    runnable: Serialized,
    tags: string[] = [],
    fallback: string,
  ): string {
    const firstTag = tags.at(0)?.toString();
    const tagName = firstTag?.startsWith('map:key') ? firstTag : undefined;
    const runnableName = runnable.id.at(-1)?.toString();

    if (runnableName === 'RunnableRetry') {
      return 'RunnableRetry';
    }

    return runName ?? tagName ?? runnableName ?? fallback;
  }
}

export default LangfuseCallbackHandler;
