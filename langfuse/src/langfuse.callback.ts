import { BaseCallbackHandler, NewTokenIndices } from '@langchain/core/callbacks/base';
import { Serialized } from '@langchain/core/load/serializable';
import { ChainValues } from '@langchain/core/utils/types';
import {
  ChatGenerationChunk,
  Generation,
  GenerationChunk,
  LLMResult,
} from '@langchain/core/outputs';
import { LangfuseObservation, startObservation } from '@langfuse/tracing';
import { AgentAction, AgentFinish } from '@langchain/core/agents';
import { isDefined } from 'class-validator';
import { DocumentInterface } from '@langchain/core/documents';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  MessageContentComplex,
  MessageContentImageUrl,
  UsageMetadata,
} from '@langchain/core/messages';
import { cleanStackTrace, parseParentContext } from './langfuse.utils';
import { circularTransformer } from './langfuse.parser';
import { serializeInputsOutputs } from './langfuse.serializer';
import { uploadMediaToLangfuse } from './langfuse.media';
import { LangfuseMedia } from '@langfuse/core';

export interface LangfuseCallbackOptions {
  parent?: LangfuseObservation;
}

export type LlmMessage = {
  role?: string;
  content?: string | Record<string, any> | Array<Record<string, any>>;
  additional_kwargs?: Record<string, any>;
};

type ParentRunId = string;

export class LangfuseCallbackHandler extends BaseCallbackHandler {
  name = this.constructor.name;

  public root?: LangfuseObservation = undefined;
  private observationMap: Record<string, LangfuseObservation | ParentRunId> = {};
  private completionStartTimes: Record<string, Date> = {};

  constructor(options?: LangfuseCallbackOptions) {
    super();
    this.root = options?.parent;
  }

  // ============================================================================
  // Helper Methods
  // ============================================================================

  private extractModelParameters(
    extraParams?: Record<string, unknown>,
  ): Record<string, string | number> | undefined {
    if (!extraParams?.invocation_params || typeof extraParams.invocation_params !== 'object') {
      return undefined;
    }

    const entries = Object.entries(extraParams.invocation_params)
      .map(([k, v]) => {
        if (typeof v === 'string' || typeof v === 'number') {
          return [k, v];
        }
      })
      .filter(isDefined);

    return Object.fromEntries(entries);
  }

  private createMetadata(
    _metadata: Record<string, unknown> | undefined,
    langchain: Serialized,
    tags?: string[],
  ): Record<string, unknown> {
    return { ..._metadata, langchain, tags };
  }

  private initializeRootIfNeeded(
    observation: LangfuseObservation,
    runName: string,
    input: any,
    metadata: Record<string, unknown>,
    tags?: string[],
  ): void {
    if (this.root) return;
    this.root = observation;
    observation.updateTrace({ tags, input, metadata, name: runName });
  }

  private registerObservation(runId: string, observation: LangfuseObservation): void {
    this.observationMap[runId] = observation;
  }

  // ============================================================================
  // Handler Methods
  // ============================================================================

  handleChatModelStart = async (
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    _metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<any> => {
    const runName = this.getName(name, llm, tags, 'LangChain Chat Model');
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    const metadata = this.createMetadata(_metadata, llm, tags);
    const parent = this.getParent(parentRunId);
    const modelParameters = this.extractModelParameters(extraParams);

    const observation = startObservation(
      runName,
      { metadata, modelParameters },
      { asType: 'generation', parentSpanContext: parseParentContext(parent) },
    );
    const isMessageImageContent = (obj: MessageContentComplex): obj is MessageContentImageUrl =>
      obj.type === 'image_url';

    const prompts: typeof messages = (await circularTransformer(
      await serializeInputsOutputs(messages),
      [
        async (message) => {
          if (!(message instanceof BaseMessage)) return;
          if (!Array.isArray(message.content)) return;

          const promises = message.content.map(async (content) => {
            if (!isMessageImageContent(content)) return;

            const image_url =
              typeof content.image_url === 'string' ? content.image_url : content.image_url.url;
            if (!image_url.startsWith('data:')) return;

            const media = new LangfuseMedia({
              source: 'base64_data_uri',
              base64DataUri: image_url,
            });

            const tag = await media.getTag();
            if (tag) {
              if (typeof content.image_url === 'string') {
                content.image_url = tag;
              } else {
                content.image_url.url = tag;
              }
            }
            void uploadMediaToLangfuse(media, 'input', observation.traceId, observation.id);
          });

          await Promise.all(promises);
        },
      ],
    )) as any;

    observation.update({ input: prompts });

    this.initializeRootIfNeeded(observation, runName, prompts, metadata, tags);
    this.registerObservation(runId, observation);
  };

  handleLLMStart = async (
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    _metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> => {
    const runName = this.getName(name, llm, tags, 'LangChain LLM');
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    const metadata = this.createMetadata(_metadata, llm, tags);
    const input = await serializeInputsOutputs(prompts);
    const parent = this.getParent(parentRunId);
    const modelParameters = this.extractModelParameters(extraParams);

    const observation = startObservation(
      runName,
      { input, metadata, modelParameters },
      { asType: 'generation', parentSpanContext: parseParentContext(parent) },
    );

    this.initializeRootIfNeeded(observation, runName, input, metadata, tags);
    this.registerObservation(runId, observation);
  };

  handleChainStart = async (
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    _metadata?: Record<string, unknown>,
    runType?: string,
    name?: string,
  ): Promise<void> => {
    const runName = this.getName(name, chain, tags, 'LangChain Run');
    if (!this.shouldTrace(runName, runId, parentRunId, tags)) return;

    const metadata = this.createMetadata(_metadata, chain, tags);
    const input = await serializeInputsOutputs(inputs);
    const parent = this.getParent(parentRunId);

    const observation = this.createChainObservation(runName, input, metadata, parent, runType);

    this.initializeRootIfNeeded(observation, runName, input, metadata, tags);
    this.registerObservation(runId, observation);
  };

  private createChainObservation(
    runName: string,
    input: any,
    metadata: Record<string, unknown>,
    parent: LangfuseObservation | undefined,
    runType?: string,
  ): LangfuseObservation {
    const parentSpanContext = parseParentContext(parent);

    if (runType === 'agent') {
      return startObservation(runName, { input, metadata }, { asType: 'agent', parentSpanContext });
    }
    if (runType === 'parser') {
      return startObservation(
        runName,
        { input, metadata, level: 'DEBUG' },
        { asType: 'guardrail', parentSpanContext },
      );
    }
    if (runType === 'prompt') {
      return startObservation(
        runName,
        { input, metadata, level: 'DEBUG' },
        { asType: 'span', parentSpanContext },
      );
    }

    return startObservation(runName, { input, metadata }, { asType: 'chain', parentSpanContext });
  }

  handleAgentAction = async (
    action: AgentAction,
    runId: string,
    parentRunId?: string,
    tags?: string[],
  ): Promise<void> => {
    const runName = this.getName(
      undefined,
      { lc: 1, type: 'not_implemented', id: [] },
      tags,
      action.tool,
    );
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    const parent = this.getParent(parentRunId);
    const input = serializeInputsOutputs(
      typeof action.toolInput === 'string' ? JSON.parse(action.toolInput) : action.toolInput,
    );

    const observation = startObservation(
      runName,
      { input },
      { asType: 'tool', parentSpanContext: parseParentContext(parent) },
    );

    this.initializeRootIfNeeded(observation, runName, input, {}, tags);
    this.registerObservation(runId, observation);
  };

  handleRetrieverStart = async (
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    _metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> => {
    const runName = this.getName(name, retriever, tags, 'LangChain Retriever');
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    const metadata = this.createMetadata(_metadata, retriever, tags);
    const input = await serializeInputsOutputs(query);
    const parent = this.getParent(parentRunId);

    const observation = startObservation(
      runName,
      { input, metadata },
      { asType: 'retriever', parentSpanContext: parseParentContext(parent) },
    );

    this.initializeRootIfNeeded(observation, runName, input, metadata, tags);
    this.registerObservation(runId, observation);
  };

  handleToolStart = async (
    tool: Serialized,
    _input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    _metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<any> => {
    const runName = this.getName(name, tool, tags, 'LangChain execution');
    if (!this.shouldTrace(runName, runId, parentRunId)) return;

    const metadata = this.createMetadata(_metadata, tool, tags);
    const input = await serializeInputsOutputs(_input);
    const parent = this.getParent(parentRunId);

    const observation = startObservation(
      runName,
      { input, metadata },
      { asType: 'tool', parentSpanContext: parseParentContext(parent) },
    );

    this.initializeRootIfNeeded(observation, runName, input, metadata, tags);
    this.registerObservation(runId, observation);
  };

  handleLLMNewToken(_token: string, _idx: NewTokenIndices, runId: string): any {
    if (runId && !(runId in this.completionStartTimes)) {
      this.completionStartTimes[runId] = new Date();
    }
  }

  handleChainEnd = (outputs: ChainValues, runId: string) => this.handleEnd(outputs, runId);
  handleChainError = (err: Error, runId: string) => this.handleError(err, runId);
  handleLLMEnd = async (output: LLMResult, runId: string) => {
    const observation = this.getObservation(runId);
    if (!observation) return;
    if ('update' in observation) {
      const lastResponse =
        output.generations[output.generations.length - 1][
          output.generations[output.generations.length - 1].length - 1
        ];
      const llmUsage = this.extractUsageMetadata(lastResponse) ?? output.llmOutput?.['tokenUsage'];
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
        'message' in lastResponse && lastResponse['message'] instanceof BaseMessage
          ? await serializeInputsOutputs(lastResponse['message'])
          : lastResponse.text;

      observation.update({
        output: extractedOutput,
        model: modelName,
        completionStartTime: this.completionStartTimes[runId],
        usageDetails,
      });
    }
    observation.end();
  };
  handleLLMError = (err: Error, runId: string) => this.handleError(err, runId);
  handleAgentEnd = (action: AgentFinish, runId: string) => this.handleEnd(action, runId);
  handleRetrieverEnd = (documents: DocumentInterface[], runId: string) =>
    this.handleEnd(documents, runId);
  handleRetrieverError = (err: Error, runId: string) => this.handleError(err, runId);
  handleToolEnd = (output: string, runId: string) => this.handleEnd(output, runId);
  handleToolError = (err: Error, runId: string) => this.handleError(err, runId);

  private getParent = (parentRunId: string | undefined): LangfuseObservation | undefined => {
    const observation = this.getObservation(parentRunId);
    return parentRunId ? observation : this.root;
  };

  private getObservation = (runId?: string): LangfuseObservation | undefined => {
    if (!runId) return;

    const recursiveGet = (runId: string): LangfuseObservation | undefined => {
      const item = this.observationMap[runId];
      if (typeof item === 'string') {
        return recursiveGet(item);
      }
      return item;
    };

    return recursiveGet(runId);
  };

  private handleEnd = async (output: any, runId: string): Promise<void> => {
    const observation = this.getObservation(runId);
    if (!observation) return;
    if ('update' in observation) {
      observation.update({ output: await serializeInputsOutputs(output) });
    }
    observation.end();
  };

  private handleError = (err: Error, runId: string) => {
    const observation = this.getObservation(runId);
    if (!observation) return;
    if ('update' in observation) {
      observation.update({
        statusMessage: cleanStackTrace(err?.stack) ?? err?.message ?? 'Unknown Error',
        level: 'ERROR',
      });
    }
    observation.end();
  };

  private extractUsageMetadata(generation: Generation): UsageMetadata | undefined {
    try {
      if (
        'message' in generation &&
        (generation['message'] instanceof AIMessage ||
          generation['message'] instanceof AIMessageChunk)
      ) {
        return generation['message'].usage_metadata;
      } else {
        return undefined;
      }
    } catch {}
  }

  private extractModelNameFromMetadata(generation: any): string | undefined {
    if (generation instanceof ChatGenerationChunk) {
      return (
        generation.generationInfo?.model_name ??
        (generation.message.response_metadata as any).model_name
      );
    }
    if (generation instanceof GenerationChunk) {
      return generation.generationInfo?.model_name;
    }
    try {
      return 'message' in generation &&
        (generation['message'] instanceof AIMessage ||
          generation['message'] instanceof AIMessageChunk)
        ? generation['message'].response_metadata.model_name
        : undefined;
    } catch {}
  }

  private shouldTrace = (runName: string, runId: string, parentRunId?: string, tags?: string[]) => {
    if (!parentRunId) return true;
    if (runName.startsWith('Runnable')) {
      this.observationMap[runId] = parentRunId;
      return false;
    }
    return true;
  };

  private getName = (
    runName: string | undefined,
    runnable: Serialized,
    tags: string[] = [],
    fallback: string,
  ) => {
    const firstTag = tags.at(0)?.toString();
    const tagName = firstTag?.startsWith('map:key') ? firstTag : undefined;
    const runnableName = runnable.id.at(-1)?.toString();

    // Forcing RunnableRetry Naming to exclude from tracing
    if (runnableName === 'RunnableRetry') {
      return 'RunnableRetry';
    }

    return runName ?? tagName ?? runnableName ?? fallback;
  };
}
