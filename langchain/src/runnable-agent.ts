import { Runnable, RunnableConfig, RunnableLike, RunnableMap } from '@langchain/core/runnables';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  coerceMessageLikeToMessage,
  ContentBlock,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { BaseChatModel, BindToolsInput } from '@langchain/core/language_models/chat_models';
import { ZodSchema } from 'zod';
import { CallbackManagerForChainRun } from '@langchain/core/callbacks/manager';
import { ToolCall } from '@langchain/core/messages/tool';
import {
  BaseListChatMessageHistory,
  InMemoryChatMessageHistory,
} from '@langchain/core/chat_history';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';

export interface AgentToolCall<T extends Record<string, any> = Record<string, any>> {
  schema: ZodSchema<T>;
  runnable: RunnableLike<T, Record<string, any>>;
}

export const agentToolCall = <T extends Record<string, any> = Record<string, any>>(
  name: string,
  description: string,
  schema: ZodSchema<T>,
  runnable: RunnableLike<T, Record<string, any>>,
): AgentToolCallOptions<T> => ({ name, description, tool: { schema, runnable } });

export interface AgentToolCallOptions<T extends Record<string, any> = Record<string, any>> {
  name: string;
  description: string;
  tool: AgentToolCall<T>;
}

export interface RunnableAgentOptions {
  llm: BaseChatModel;
  tools: AgentToolCallOptions[];
  history?: BaseListChatMessageHistory;
  // End the agent using a tool_call. Ensure you instruct the AI to use this to end the agent.
  tool_response?: AgentToolCallOptions;
  maxToolCalls?: number;
}

export class RunnableAgent extends Runnable<BaseLanguageModelInput, BaseMessage[]> {
  lc_namespace = ['langchain', 'agents', 'RunnableAgent'];

  private _callsLeft = 10;

  constructor(private readonly options: RunnableAgentOptions) {
    super();
    this._callsLeft = options.maxToolCalls ?? 10;
  }

  private _cleanHistory = (history: BaseMessage[]): BaseMessage[] => {
    return history.filter((message) => {
      if (AIMessage.isInstance(message) && typeof message.content !== 'string') {
        message.content = message.content.filter((content) => {
          if (content.type === 'text') {
            return (content as ContentBlock.Text).text.trim().length > 0;
          }
          return true;
        });
      }
      return true;
    });
  };

  private _createToolResponse = (
    tool_call: ToolCall,
    toolResponse: Record<string, any>,
    status: 'success' | 'error' = 'success',
  ): BaseMessage => {
    return new ToolMessage({
      content: JSON.stringify(toolResponse),
      name: tool_call.name,
      tool_call_id: tool_call.id ?? '',
      status: status,
    });
  };

  private runToolCallingAgent = async (
    input: BaseMessage[],
    config?: Partial<RunnableConfig>,
    runManager?: CallbackManagerForChainRun,
  ) => {
    if (this._callsLeft === 0) {
      throw new Error('Max tool calls reached');
    }
    this._callsLeft--;

    const tools: BindToolsInput[] = this.options.tools
      .map(
        (toolDef): BindToolsInput => ({
          name: toolDef.name,
          description: toolDef.description,
          schema: toolDef.tool.schema,
        }),
      )
      // Add ToolResponse if provided
      .concat(
        this.options.tool_response
          ? [
              {
                name: this.options.tool_response.name,
                description: this.options.tool_response.description,
                schema: this.options.tool_response.tool.schema,
              },
            ]
          : [],
      );

    const boundLlm = this.options.llm.bindTools?.(tools, {
      tool_choice: this.options.tool_response ? 'any' : 'auto',
    });

    if (!boundLlm) {
      throw new Error('LLM does not support tool binding');
    }

    return boundLlm
      .withConfig({
        ...config,
        callbacks: runManager?.getChild('llm'),
      })
      .stream(this._cleanHistory(input));
  };

  private runToolCallMessage = async (
    input: AIMessage,
    config?: Partial<RunnableConfig>,
    runManager?: CallbackManagerForChainRun,
  ): Promise<BaseMessage[]> => {
    const promises = (input.tool_calls ?? []).map(
      async (tool_call: ToolCall): Promise<BaseMessage> => {
        const toolResponse = this.options.tool_response;
        const findTool = [...this.options.tools, ...(toolResponse ? [toolResponse] : [])].find(
          (tool) => tool.name === tool_call.name,
        );

        const callbackHandler = await runManager?.getChild('tool').handleToolStart(
          { name: tool_call.name, type: 'not_implemented', lc: 0, id: [] },
          JSON.stringify(tool_call.args),
          undefined,
          config?.runId,
          ['tool_call'],
          {
            name: findTool?.name,
            description: findTool?.description,
          },
          tool_call.name,
        );

        if (!findTool) {
          await callbackHandler?.handleToolError(
            new Error(`Tool "${tool_call.name}" was not found.`),
          );
          return this._createToolResponse(
            tool_call,
            {
              type: 'text',
              text: JSON.stringify({
                success: false,
                error: `Tool "${tool_call.name}" was not found.`,
              }),
            },
            'error',
          );
        }

        const zodSchema = findTool.tool.schema;
        const toolRunnable = findTool.tool.runnable;
        let result = {};

        try {
          if (Runnable.isRunnable(toolRunnable)) {
            result = await toolRunnable.invoke(zodSchema.parse(tool_call.args), {
              ...config,
              callbacks: callbackHandler?.getChild(),
            });
          } else {
            if (typeof toolRunnable === 'function') {
              result = await toolRunnable(zodSchema.parse(tool_call.args), {
                ...config,
                callbacks: callbackHandler?.getChild(),
              });
            } else {
              result = await RunnableMap.from(toolRunnable as any)
                .withConfig({ callbacks: callbackHandler?.getChild('chain') })
                .invoke(zodSchema.parse(tool_call.args));
            }
          }
        } catch (e) {
          await callbackHandler?.handleToolError(
            new Error(`Tool "${tool_call.name}" was not found.`),
          );
          const errorResponse =
            e instanceof Error ? { error: e.message } : (e as Record<string, any>);
          return this._createToolResponse(tool_call, errorResponse, 'error');
        }

        await callbackHandler?.handleToolEnd(JSON.stringify(result));

        return this._createToolResponse(tool_call, result);
      },
    );

    return await Promise.all(promises);
  };

  async invoke(input: string, options?: Partial<RunnableConfig>): Promise<BaseMessage[]> {
    const stream = await this.stream(input, options);
    let response: BaseMessage[] = [];
    for await (const chunk of stream) {
      response = chunk;
    }
    return response;
  }

  private _parseModelInputToBaseMessage = (input: BaseLanguageModelInput): BaseMessage[] => {
    if (typeof input === 'string') {
      return [new HumanMessage(input)];
    }

    if (Array.isArray(input)) {
      return input.map((item) => coerceMessageLikeToMessage(item));
    }

    return input.toChatMessages();
  };

  async stream(
    input: BaseLanguageModelInput,
    options?: Partial<RunnableConfig>,
  ): Promise<IterableReadableStream<BaseMessage[]>> {
    const self = this;

    async function* generator() {
      yield input;
    }

    const returnGenerator = this._transformStreamWithConfig(
      generator(),
      async function* (_, runManager, config) {
        // Store Initial History
        const initialHistory = (await self.options.history?.getMessages()) ?? [];
        const sessionHistory = new InMemoryChatMessageHistory();

        // Add Human input provided history
        await self.options.history?.addMessages(self._parseModelInputToBaseMessage(input));

        while (true) {
          // Consolidate history and invoke LLM
          const sessionMessages = await sessionHistory.getMessages();
          const messages = [
            ...initialHistory,
            ...self._parseModelInputToBaseMessage(input),
            ...sessionMessages,
          ];
          const agentStream = await self.runToolCallingAgent(messages, config, runManager);

          let agentResponse = new AIMessageChunk('');
          let isToolCall = false;
          for await (const chunk of agentStream) {
            agentResponse = agentResponse.concat(chunk);

            // Check if one of the chunks is a tool_call
            if (!isToolCall && agentResponse.tool_calls?.length) {
              isToolCall = true;
            }

            // If we determined this is a streamed tool_call, only stream if the tool_call data is provided.
            // This is done because as it is streaming, tool_call can go undefined, because it is building the args.
            // This is so the args and tool_call is always valid so there is no flickering in tool_call data.
            if (isToolCall) {
              if (agentResponse.tool_calls?.length) {
                yield [...sessionMessages, agentResponse];
              }
            } else {
              yield [...sessionMessages, agentResponse];
            }
          }

          // Add AI Response message to history
          await self.options.history?.addMessage(agentResponse);
          await sessionHistory.addMessage(agentResponse);

          // Check if tool_call or final response
          if (isToolCall) {
            const toolResponse = await self.runToolCallMessage(agentResponse, config, runManager);
            await self.options.history?.addMessages(toolResponse);
            await sessionHistory.addMessages(toolResponse);

            // Check if user wants a tool_call as a stop
            if (self.options.tool_response) {
              const toolStopCalled = agentResponse.tool_calls?.some(
                (tool_call) => tool_call.name === self.options.tool_response!.name,
              );

              if (toolStopCalled) break;
            }
          } else {
            break;
          }
        }

        return sessionHistory.getMessages();
      },
      { ...options, runType: 'chain' },
    );

    return IterableReadableStream.fromAsyncGenerator(returnGenerator);
  }
}
