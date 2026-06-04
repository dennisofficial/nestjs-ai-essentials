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
  LangfuseEvent,
  LangfuseGenerationAttributes,
  LangfuseObservation,
  LangfuseSpanAttributes,
  propagateAttributes,
  startActiveObservation,
} from '@langfuse/tracing';

const LANGSMITH_HIDDEN_TAG = 'langsmith:hidden';

type ObservationType =
  | 'agent'
  | 'chain'
  | 'generation'
  | 'guardrail'
  | 'retriever'
  | 'span'
  | 'tool';

type LangfusePrompt = {
  name: string;
  version: number;
  isFallback: boolean;
};

/*
A prompt can be registered either directly, or as a thunk that resolves it lazily.
The lazy form lets a runnable attach `metadata.langfusePrompt` at construction time
(before it knows which prompt version it will use) and populate it mid-run — the thunk
is evaluated when the generation links to it, by which point it has resolved.
*/
type LangfusePromptResolver = LangfusePrompt | (() => LangfusePrompt | undefined);

export type LlmMessage = {
  role: string;
  content: BaseMessageFields['content'];
  additional_kwargs?: BaseMessageFields['additional_kwargs'];
};

export type AnonymousLlmMessage = {
  content: BaseMessageFields['content'];
  additional_kwargs?: BaseMessageFields['additional_kwargs'];
};

export interface LangfuseCallbackOptions {
  parent?: LangfuseObservation;
  userId?: string;
  sessionId?: string;
  tags?: string[];
  version?: string;
  traceMetadata?: Record<string, unknown>;
}

export class LangfuseCallbackHandler extends BaseCallbackHandler {
  name = 'LangfuseCallbackHandler';

  private readonly userId?: string;
  private readonly version?: string;
  private readonly sessionId?: string;
  private readonly tags: string[];
  private readonly traceMetadata?: Record<string, unknown>;
  private readonly parentObserver?: LangfuseObservation;

  private completionStartTimes: Record<string, Date> = {};
  private promptToParentRunMap = new Map<string, LangfusePromptResolver>();
  private runMap: Map<string, LangfuseObservation> = new Map();
  private skippedRunMap: Map<string, string> = new Map();

  public last_trace_id: string | null = null;

  constructor(params?: LangfuseCallbackOptions) {
    super({});

    this.sessionId = params?.sessionId;
    this.userId = params?.userId;
    this.tags = params?.tags ?? [];
    this.traceMetadata = params?.traceMetadata;
    this.version = params?.version;
    this.parentObserver = params?.parent;
  }

  get logger() {
    return getGlobalLogger();
  }

  async handleLLMNewToken(
    token: string,
    _idx: any,
    runId: string,
    _parentRunId?: string,
    _tags?: string[],
    _fields?: any,
  ): Promise<void> {
    // if this is the first token, add it to completionStartTimes
    if (runId && !(runId in this.completionStartTimes)) {
      this.logger.debug(`LLM first streaming token: ${runId}`);
      this.completionStartTimes[runId] = new Date();
    }
  }

  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    runType?: string,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, chain, tags, 'Langchain Run');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Chain start with Id: ${runId}`);

      this.registerLangfusePrompt(parentRunId, metadata);

      // In chains, inputs can be a string or an array of BaseMessage
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

      // If there's no parent run, this is a top-level chain execution
      // and we need to propagate trace attributes
      if (!parentRunId) {
        const traceTags = [...new Set([...(tags ?? []), ...this.tags])];

        const traceUserId =
          metadata && 'langfuseUserId' in metadata && typeof metadata['langfuseUserId'] === 'string'
            ? metadata['langfuseUserId']
            : this.userId;

        const traceSessionId =
          metadata &&
          'langfuseSessionId' in metadata &&
          typeof metadata['langfuseSessionId'] === 'string'
            ? metadata['langfuseSessionId']
            : this.sessionId;

        const traceMetadata = this.traceMetadata
          ? Object.fromEntries(
              Object.entries(this.traceMetadata).map(([k, v]) => [
                k,
                typeof v === 'string' ? v : JSON.stringify(v),
              ]),
            )
          : undefined;

        propagateAttributes(
          {
            tags: traceTags,
            userId: traceUserId,
            sessionId: traceSessionId,
            metadata: traceMetadata,
            version: this.version,
          },
          () => {
            this.startAndRegisterOtelSpan({
              runName,
              parentRunId,
              runId,
              type: this.getChainObservationType(runType),
              tags,
              metadata,
              attributes: {
                input: finalInput,
              },
            });
          },
        );
      } else {
        this.startAndRegisterOtelSpan({
          runName,
          parentRunId,
          runId,
          type: this.getChainObservationType(runType),
          tags,
          metadata,
          attributes: {
            input: finalInput,
          },
        });
      }
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleAgentAction(action: AgentAction, runId: string, parentRunId?: string): Promise<void> {
    try {
      this.logger.debug(`Agent action ${action.tool} with ID: ${runId}`);
      this.startAndRegisterOtelSpan({
        runId,
        parentRunId,
        type: 'tool',
        runName: action.tool,
        attributes: {
          input: action,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleAgentEnd?(
    action: AgentFinish,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string,
  ): Promise<void> {
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

  async handleChainError(
    err: any,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      this.logger.debug(`Chain error: ${err} with ID: ${runId}`);

      const azureRefusalError = this.parseAzureRefusalError(err);

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage: err.toString() + azureRefusalError,
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
    parentRunId?: string | undefined,
    extraParams?: Record<string, unknown> | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    const runName = this.getName(name, llm, tags, 'Langchain Generation');
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    this.logger.debug(`Generation start with ID: ${runId} and parentRunId ${parentRunId}`);

    const modelParameters: Record<string, any> = {};
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
      _type?: string;
      model?: string;
      model_name?: string;
      repo_id?: string;
    }

    let extractedModelName: string | undefined;
    if (extraParams) {
      const invocationParamsModelName = (extraParams.invocation_params as InvocationParams).model;
      const metadataModelName =
        metadata && 'ls_model_name' in metadata ? (metadata['ls_model_name'] as string) : undefined;

      extractedModelName = invocationParamsModelName ?? metadataModelName;
    }

    const { prompt: registeredPrompt, runId: registeredPromptRunId } =
      this.findRegisteredPrompt(parentRunId);

    if (registeredPrompt && registeredPromptRunId) {
      this.deregisterLangfusePrompt(registeredPromptRunId);
    }

    this.startAndRegisterOtelSpan({
      type: 'generation',
      runId,
      parentRunId,
      metadata,
      tags,
      runName,
      attributes: {
        input: messages,
        model: extractedModelName,
        modelParameters: modelParameters,
        prompt: registeredPrompt,
      },
    });
  }

  async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string | undefined,
    extraParams?: Record<string, unknown> | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    try {
      this.logger.debug(`Chat model start with ID: ${runId}`);

      const prompts = messages.flatMap((message) =>
        message.map((m) => this.extractChatMessageContent(m)),
      );

      void this.handleGenerationStart(
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

  async handleChainEnd(
    outputs: ChainValues,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
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
    parentRunId?: string | undefined,
    extraParams?: Record<string, unknown> | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    try {
      this.logger.debug(`LLM start with ID: ${runId}`);

      void this.handleGenerationStart(
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
    parentRunId?: string | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, tool, tags, 'Tool execution');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Tool start with ID: ${runId}`);

      this.startAndRegisterOtelSpan({
        runId,
        parentRunId,
        type: 'tool',
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
    parentRunId?: string | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    try {
      const runName = this.getName(name, retriever, tags, 'Retriever');
      if (!this.shouldTrace(runName, runId, parentRunId)) return;

      this.logger.debug(`Retriever start with ID: ${runId}`);

      this.startAndRegisterOtelSpan({
        runId,
        parentRunId,
        type: 'retriever',
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
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
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

  async handleRetrieverError(
    err: any,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      this.logger.debug(`Retriever error: ${err} with ID: ${runId}`);
      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage: err.toString(),
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }
  async handleToolEnd(
    output: string,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
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

  async handleToolError(
    err: any,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      this.logger.debug(`Tool error ${err} with ID: ${runId}`);

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage: err.toString(),
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      this.logger.debug(`LLM end with ID: ${runId}`);

      const lastResponse =
        output.generations[output.generations.length - 1][
          output.generations[output.generations.length - 1].length - 1
        ];
      const llmUsage =
        this.extractUsageMetadata(lastResponse) ?? output.llmOutput?.['tokenUsage'] ?? {};
      const modelName = this.extractModelNameFromMetadata(lastResponse);

      const usageDetails: Record<string, any> = {
        input:
          llmUsage?.input_tokens ??
          ('promptTokens' in llmUsage ? llmUsage?.promptTokens : undefined),
        output:
          llmUsage?.output_tokens ??
          ('completionTokens' in llmUsage ? llmUsage?.completionTokens : undefined),
        total:
          llmUsage?.total_tokens ?? ('totalTokens' in llmUsage ? llmUsage?.totalTokens : undefined),
      };

      if (llmUsage && 'input_token_details' in llmUsage) {
        for (const [key, val] of Object.entries(llmUsage['input_token_details'] ?? {})) {
          usageDetails[`input_${key}`] = val;

          if ('input' in usageDetails && typeof val === 'number') {
            usageDetails['input'] = Math.max(0, usageDetails['input'] - val);
          }
        }
      }

      if (llmUsage && 'output_token_details' in llmUsage) {
        for (const [key, val] of Object.entries(llmUsage['output_token_details'] ?? {})) {
          usageDetails[`output_${key}`] = val;

          if ('output' in usageDetails && typeof val === 'number') {
            usageDetails['output'] = Math.max(0, usageDetails['output'] - val);
          }
        }
      }

      const extractedOutput =
        'message' in lastResponse
          ? this.extractChatMessageContent(lastResponse['message'] as BaseMessage)
          : lastResponse.text;

      this.handleOtelSpanEnd({
        runId,
        type: 'generation',
        attributes: {
          model: modelName,
          output: extractedOutput,
          completionStartTime:
            runId in this.completionStartTimes ? this.completionStartTimes[runId] : undefined,
          usageDetails: usageDetails,
        },
      });

      if (runId in this.completionStartTimes) {
        delete this.completionStartTimes[runId];
      }
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  async handleLLMError(
    err: any,
    runId: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      this.logger.debug(`LLM error ${err} with ID: ${runId}`);

      // Azure has the refusal status for harmful messages in the error property
      // This would not be logged as the error message is only a generic message
      // that there has been a refusal
      const azureRefusalError = this.parseAzureRefusalError(err);

      this.handleOtelSpanEnd({
        runId,
        attributes: {
          level: 'ERROR',
          statusMessage: err.toString() + azureRefusalError,
        },
      });
    } catch (e) {
      this.logger.debug(e instanceof Error ? e.message : String(e));
    }
  }

  private registerLangfusePrompt(parentRunId?: string, metadata?: Record<string, unknown>): void {
    /*
    Register a prompt for linking to a generation with the same parentRunId.

    `parentRunId` must exist when we want to do any prompt linking to a generation. If it does not exist, it means the execution is solely a Prompt template formatting without any following LLM invocation, so no generation will be created to link to.
    For the simplest chain, a parent run is always created to wrap the individual runs consisting of prompt template formatting and LLM invocation.
    So, we do not need to register any prompt for linking if parentRunId is missing.
    */
    if (metadata && 'langfusePrompt' in metadata && parentRunId) {
      this.promptToParentRunMap.set(parentRunId, metadata.langfusePrompt as LangfusePromptResolver);
    }
  }

  private deregisterLangfusePrompt(runId: string): void {
    this.promptToParentRunMap.delete(runId);
  }

  /*
  Find a prompt registered for the given run or any of its ancestors.

  A prompt is registered against the parentRunId of the prompt-template run (see `registerLangfusePrompt`). The generation that should link to it does not always share that parentRunId: it can be nested several levels deeper when the LLM is wrapped in retry/fallback runnables. Those wrappers are `Runnable*` runs that are skipped from tracing and recorded in `skippedRunMap` (see `shouldTrace`), so we climb that chain — checking each level for a registered prompt — until we find one or run out of skipped ancestors. The `visited` set guards against cycles, mirroring `resolveParentRunId`.

  Returns the prompt together with the runId it was registered under, so the caller can deregister the correct key.
  */
  private findRegisteredPrompt(parentRunId?: string): {
    prompt?: LangfusePrompt;
    runId?: string;
  } {
    let currentRunId: string | undefined = parentRunId ?? 'root';
    const visited = new Set<string>();

    while (currentRunId && !visited.has(currentRunId)) {
      visited.add(currentRunId);

      const entry = this.promptToParentRunMap.get(currentRunId);
      if (entry) {
        // Resolve the thunk form lazily; a direct prompt passes through unchanged.
        const prompt = typeof entry === 'function' ? entry() : entry;
        if (prompt) {
          return { prompt, runId: currentRunId };
        }
      }

      currentRunId = this.skippedRunMap.get(currentRunId);
    }

    return {};
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
    const { type, runName, runId, parentRunId, attributes, metadata, tags } = params;
    const observationAttributes = {
      version: this.version,
      metadata: this.joinTagsAndMetaData(tags, metadata),
      level: tags && tags.includes(LANGSMITH_HIDDEN_TAG) ? 'DEBUG' : undefined,
      ...attributes,
    } as LangfuseGenerationAttributes;

    const observation = startActiveObservation(
      runName,
      (gen) => {
        gen.update(observationAttributes);
        return gen;
      },
      {
        // @ts-ignore
        asType: type ?? 'span',
        parentSpanContext: this.getParentSpanContext(parentRunId),
        endOnExit: false,
      },
    ) as LangfuseObservation;
    this.runMap.set(runId, observation);

    return observation;
  }

  private handleOtelSpanEnd(params: {
    runId: string;
    attributes?: LangfuseSpanAttributes;
    type?: 'span';
  }): void;
  private handleOtelSpanEnd(params: {
    runId: string;
    attributes?: LangfuseGenerationAttributes;
    type: 'generation';
  }): void;
  private handleOtelSpanEnd(params: {
    runId: string;
    attributes?: LangfuseGenerationAttributes | LangfuseSpanAttributes;
    type?: 'span' | 'generation';
  }) {
    const { runId, attributes = {} } = params;

    const span = this.runMap.get(runId);
    if (!span) {
      return;
    }

    if (!(span instanceof LangfuseEvent)) {
      span.update(attributes).end();
    }

    this.last_trace_id = span.traceId;
    this.runMap.delete(runId);
  }
  private parseAzureRefusalError(err: any): string {
    // Azure has the refusal status for harmful messages in the error property
    // This would not be logged as the error message is only a generic message
    // that there has been a refusal
    let azureRefusalError = '';
    if (typeof err == 'object' && 'error' in err) {
      try {
        azureRefusalError = '\n\nError details:\n' + JSON.stringify(err['error'], null, 2);
      } catch {}
    }

    return azureRefusalError;
  }

  private joinTagsAndMetaData(
    tags?: string[] | undefined,
    metadata1?: Record<string, unknown> | undefined,
    metadata2?: Record<string, unknown> | undefined,
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
    if (!metadata) {
      return;
    }

    const langfuseKeys = ['langfusePrompt', 'langfuseUserId', 'langfuseSessionId'];

    return Object.fromEntries(
      Object.entries(metadata).filter(([key, _]) => !langfuseKeys.includes(key)),
    );
  }

  /** Not all models supports tokenUsage in llmOutput, can use AIMessage.usage_metadata instead */
  private extractUsageMetadata(generation: Generation): UsageMetadata | undefined {
    try {
      return 'message' in generation &&
        (generation['message'] instanceof AIMessage ||
          generation['message'] instanceof AIMessageChunk)
        ? generation['message'].usage_metadata
        : undefined;
    } catch (err) {
      this.logger.debug(`Error extracting usage metadata: ${err}`);

      return;
    }
  }

  private extractModelNameFromMetadata(generation: any): string | undefined {
    try {
      return 'message' in generation &&
        (generation['message'] instanceof AIMessage ||
          generation['message'] instanceof AIMessageChunk)
        ? generation['message'].response_metadata.model_name
        : undefined;
    } catch {}
  }

  private extractChatMessageContent(
    message: BaseMessage,
  ): LlmMessage | AnonymousLlmMessage | MessageContent {
    let response = undefined;

    if (message.getType() === 'human') {
      response = { content: message.content, role: 'user' };
    } else if (message.getType() === 'generic') {
      response = {
        content: message.content,
        role: 'human',
      };
    } else if (message.getType() === 'ai') {
      response = { content: message.content, role: 'assistant' };

      if (
        'tool_calls' in message &&
        Array.isArray(message.tool_calls) &&
        (message.tool_calls?.length ?? 0) > 0
      ) {
        (response as any)['tool_calls'] = message['tool_calls'];
      }
      if ('additional_kwargs' in message && 'tool_calls' in message['additional_kwargs']) {
        (response as any)['tool_calls'] = message['additional_kwargs']['tool_calls'];
      }
    } else if (message.getType() === 'system') {
      response = { content: message.content, role: 'system' };
    } else if (message.getType() === 'function') {
      response = {
        content: message.content,
        additional_kwargs: message.additional_kwargs,
        role: message.name,
      };
    } else if (message.getType() === 'tool') {
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
      (response as any)['tool_calls'] === undefined
    ) {
      return { ...response, additional_kwargs: message.additional_kwargs };
    }

    return response;
  }

  private getParentSpanContext(parentRunId?: string) {
    const resolvedParentRunId = this.resolveParentRunId(parentRunId);
    if (resolvedParentRunId) {
      return this.runMap.get(resolvedParentRunId)?.otelSpan.spanContext();
    }
    return this.parentObserver?.otelSpan.spanContext();
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

  private getChainObservationType(runType?: string): ObservationType {
    if (runType === 'agent') return 'agent';
    if (runType === 'parser') return 'guardrail';
    if (runType === 'prompt') return 'span';
    return 'chain';
  }
}
