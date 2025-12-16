import { InMemoryChatMessageHistory } from '@langchain/core/chat_history';
import {
  Runnable,
  RunnableConfig,
  RunnableLambda,
  RunnableSequence,
} from '@langchain/core/runnables';
import { ChatPromptValueInterface } from '@langchain/core/prompt_values';
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  FunctionMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { BaseLanguageModelInput } from '@langchain/core/language_models/base';

// Re-export for backwards compatibility
export { InMemoryChatMessageHistory, InMemoryChatMessageHistory as ChatMessageHistory };

export class RunnableHistory {
  static from = <RunInput, RunOutput>(
    func: (
      input: RunInput,
      history: InMemoryChatMessageHistory,
      options?: Partial<RunnableConfig>,
    ) =>
      | RunOutput
      | Promise<RunOutput>
      | Runnable<RunInput, RunOutput>
      | Promise<Runnable<RunInput, RunOutput>>,
  ) => {
    return RunnableLambda.from<RunInput, RunOutput>(async (input, options) =>
      func(input, new InMemoryChatMessageHistory(), options),
    ).withConfig({ runName: 'RunnableHistory' });
  };

  static runPrompt = <Input extends Record<string, any>>(
    chatHistory: InMemoryChatMessageHistory,
    runnable: Runnable<Input, ChatPromptValueInterface>,
  ) =>
    RunnableSequence.from<Input, BaseMessage[]>([
      runnable,
      RunnableLambda.from<ChatPromptValueInterface, BaseMessage[]>(async (input) => {
        await chatHistory.addMessages(input.messages);
        return await chatHistory.getMessages();
      }),
    ]);

  static runLLM = <Input, Outpu>(
    chatHistory: InMemoryChatMessageHistory,
    runnable: Runnable<BaseLanguageModelInput, AIMessageChunk>,
  ) =>
    RunnableSequence.from<BaseLanguageModelInput, AIMessageChunk>([
      runnable,
      RunnableLambda.from<AIMessageChunk, AIMessageChunk>(async (input) => {
        const messages: BaseMessage[] =
          input.tool_calls
            ?.map((tool) => {
              return [
                new AIMessage({ content: '', tool_calls: [tool] }),
                new ToolMessage(JSON.stringify({ success: true }), tool.id!, tool.name),
              ];
            })
            .flat() ?? [];
        await chatHistory.addMessages(messages);
        return input;
      }),
    ]);

  /** @deprecated - Use `runLLM`, but ensure you are using the new ClassOutputFunctionParser */
  static runClaude = (
    chatHistory: InMemoryChatMessageHistory,
    runnable: Runnable<BaseLanguageModelInput, AIMessageChunk>,
  ) =>
    RunnableSequence.from<BaseLanguageModelInput, AIMessageChunk>([
      runnable,
      RunnableLambda.from<AIMessageChunk, AIMessageChunk>(async (input) => {
        const messages: BaseMessage[] =
          input.tool_calls
            ?.map((tool) => {
              return [
                new AIMessage({ content: '', tool_calls: [tool] }),
                new ToolMessage(JSON.stringify({ success: true }), tool.id!, tool.name),
              ];
            })
            .flat() ?? [];
        await chatHistory.addMessages(messages);
        return input;
      }),
    ]);

  /** @deprecated - Use `runLLM`, but ensure you are using the new ClassOutputFunctionParser */
  static runOpenAi = (
    chatHistory: InMemoryChatMessageHistory,
    runnable: Runnable<BaseLanguageModelInput, AIMessageChunk>,
  ) =>
    RunnableSequence.from<BaseLanguageModelInput, AIMessageChunk>([
      runnable,
      RunnableLambda.from<AIMessageChunk, AIMessageChunk>(async (input) => {
        const functionCalls = input.tool_calls ?? [];
        await chatHistory.addMessage(input);
        await Promise.all(
          functionCalls?.map(async (functionCall) => {
            await chatHistory.addMessage(
              new FunctionMessage({
                content: JSON.stringify({ success: true }),
                name: functionCall.name,
              }),
            );
          }),
        );
        return input;
      }),
    ]);
}
