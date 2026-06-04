import type { ChatPromptValue } from '@langchain/core/prompt_values';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Runnable, RunnableLambda } from '@langchain/core/runnables';
import { LangfuseClient } from '@langfuse/client';

type ResolvedPrompt = Awaited<ReturnType<LangfuseClient['prompt']['get']>>;

export interface LangfusePromptOptions {
  /** Fetch a specific label (e.g. `production`, `latest`). Mirrors `prompt.get`. */
  label?: string;
  /** Fetch a specific version number. Mirrors `prompt.get`. Takes precedence over `label`. */
  version?: number;
  /** Defaults to a process-wide `new LangfuseClient()`; inject one for tests/DI. */
  client?: LangfuseClient;
}

/**
 * Wraps a Langfuse-managed prompt as a drop-in step for a `RunnableSequence`.
 *
 * Declare it once next to the chain's other constants:
 *
 *   const prompt = new LangfusePrompt<PromptInput>('example_prompt', { label });
 *
 * then drop `prompt.getRunnable()` into the sequence in place of the manual
 * fetch / `getLangchainPrompt()` / `ChatPromptTemplate.fromMessages(...)` block.
 *
 * The prompt is fetched lazily on the first invocation and memoized, so building
 * the chain stays synchronous and Langfuse is hit once per process rather than per
 * call. The resolved prompt is still linked to its generation in the trace (see
 * `getRunnable`).
 */
export class LangfusePrompt<TInput extends Record<string, unknown> = Record<string, unknown>> {
  private readonly client: LangfuseClient;
  private resolved?: Promise<{ template: ChatPromptTemplate; prompt: ResolvedPrompt }>;

  constructor(
    private readonly name: string,
    private readonly options: LangfusePromptOptions = {},
  ) {
    this.client = options.client ?? new LangfuseClient();
  }

  /**
   * A runnable that formats `TInput` into a `ChatPromptValue` for the downstream LLM.
   *
   * Trace linking: the custom `LangfuseCallbackHandler` links a prompt to its
   * generation via `metadata.langfusePrompt`, registered when this runnable starts —
   * but the resolved prompt is only known mid-invoke. We therefore pass a thunk
   * (`() => holder.current`) that the handler evaluates later, at generation start,
   * once the body below has populated `holder`.
   */
  getRunnable(): Runnable<TInput, ChatPromptValue> {
    const holder: { current?: ResolvedPrompt } = {};

    return RunnableLambda.from<TInput, ChatPromptValue>(async (input, config) => {
      const { template, prompt } = await this.resolve();
      holder.current = prompt;
      return template.invoke(input, config);
    }).withConfig({
      runName: `LangfusePrompt:${this.name}`,
      metadata: { langfusePrompt: () => holder.current },
    });
  }

  private resolve(): Promise<{ template: ChatPromptTemplate; prompt: ResolvedPrompt }> {
    // Memoize the resolution; on failure, clear it so the next invoke retries.
    if (!this.resolved) {
      this.resolved = this.fetchAndBuild().catch((err) => {
        this.resolved = undefined;
        throw err;
      });
    }
    return this.resolved;
  }

  private async fetchAndBuild(): Promise<{ template: ChatPromptTemplate; prompt: ResolvedPrompt }> {
    const { label, version } = this.options;
    const prompt = await this.client.prompt.get(this.name, {
      type: 'chat',
      ...(version !== undefined ? { version } : label !== undefined ? { label } : {}),
    });

    const messages = prompt.getLangchainPrompt() as unknown as Array<{
      role: string;
      content: string;
    }>;
    const template = ChatPromptTemplate.fromMessages(
      messages.map((msg) => [msg.role, msg.content] as [string, string]),
    );

    return { template, prompt };
  }
}
